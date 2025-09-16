# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import tempfile
import pandas as pd
import glob
from tqdm import tqdm
import logging

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample

import librosa
import soundfile as sf

def downsample_audio(wav_path, target_sr=16000):
    """
    Downsample audio file to target sample rate.
    
    Args:
        wav_path (str): Path to input audio file
        target_sr (int): Target sample rate (default: 16000)
    
    Returns:
        str: Path to downsampled audio file (temporary file)
    """
    try:
        # Load audio file with original sample rate
        audio, orig_sr = librosa.load(wav_path, sr=None)
        
        # Check if downsampling is needed
        if orig_sr == target_sr:
            return wav_path
        
        # Resample to target sample rate
        audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        
        # Create temporary file for downsampled audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        # Save downsampled audio
        sf.write(temp_path, audio_resampled, target_sr)
        
        return temp_path
        
    except Exception as e:
        logging.error(f"Error during downsampling {wav_path}: {e}")
        return None

def process_wav_dataset(input_dir, output_csv, prompt_text, cfg, model, wav_processor):
    """
    Process all .wav files in a directory and save results to CSV.
    
    Args:
        input_dir (str): Directory containing .wav files
        output_csv (str): Output CSV file path
        prompt_text (str): Prompt to use for all files
        cfg: Configuration object
        model: SALMONN model
        wav_processor: Whisper feature extractor
    """
    
    # Find all .wav files
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    wav_files.extend(glob.glob(os.path.join(input_dir, "**/*.WAV"), recursive=True))
    
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    
    # Prepare results list
    results = []
    temp_files = []
    
    # Process each file
    for wav_path in tqdm(wav_files, desc="Processing audio files"):
        try:
            # Get relative path for cleaner output
            rel_path = os.path.relpath(wav_path, input_dir)
            
            # Downsample audio if needed
            processed_wav_path = downsample_audio(wav_path, target_sr=16000)
            
            if processed_wav_path is None:
                results.append({
                    'file_path': rel_path,
                    'prompt': prompt_text,
                    'output': 'ERROR: Could not process audio',
                    'status': 'failed'
                })
                continue
            
            # Keep track of temp file if it's different from original
            if processed_wav_path != wav_path:
                temp_files.append(processed_wav_path)
            
            # Prepare sample
            samples = prepare_one_sample(processed_wav_path, wav_processor)
            
            # Format prompt
            formatted_prompt = [
                cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())
            ]
            
            # Generate output
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model.generate(samples, cfg.config.generate, prompts=formatted_prompt)[0]
            
            # Clean up the output (remove special tokens, extra whitespace)
            cleaned_output = output.strip()
            if cleaned_output.startswith('<s>'):
                cleaned_output = cleaned_output[3:].strip()
            if cleaned_output.endswith('</s>'):
                cleaned_output = cleaned_output[:-4].strip()
            
            results.append({
                'file_path': rel_path,
                'prompt': prompt_text,
                'output': cleaned_output,
                'status': 'success'
            })
            
        except Exception as e:
            logging.error(f"Error processing {wav_path}: {e}")
            results.append({
                'file_path': rel_path,
                'prompt': prompt_text,
                'output': f'ERROR: {str(e)}',
                'status': 'failed'
            })
        
        # Clean up temporary file immediately after processing
        if processed_wav_path != wav_path and processed_wav_path in temp_files:
            try:
                os.unlink(processed_wav_path)
                temp_files.remove(processed_wav_path)
            except:
                pass
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Print summary
    success_count = len(df[df['status'] == 'success'])
    failed_count = len(df[df['status'] == 'failed'])
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(wav_files)} files")
    print(f"Failed: {failed_count}/{len(wav_files)} files")
    print(f"Results saved to: {output_csv}")
    
    # Clean up any remaining temporary files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Batch process WAV files with SALMONN")
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument("--input-dir", type=str, required=True, help='directory containing .wav files')
    parser.add_argument("--output-csv", type=str, required=True, help='output CSV file path')
    parser.add_argument("--prompt", type=str, required=True, help='prompt to use for all files')
    parser.add_argument("--device", type=str, default="cuda:3", help='device to use')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Loading model...")
    cfg = Config(args)
    model = SALMONN.from_config(cfg.config.model)
    model.eval()
    
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    print("Model loaded successfully!")
    
    # Process the dataset
    process_wav_dataset(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        prompt_text=args.prompt,
        cfg=cfg,
        model=model,
        wav_processor=wav_processor
    )

if __name__ == "__main__":
    main()