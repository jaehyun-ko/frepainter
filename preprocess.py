import os
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import librosa
from mel_processing import mel_spectrogram_torch

def mp_npy(npy_file: str) -> list:
    """
    Load an npy file and check if its length is within the valid range.

    Parameters:
    npy_file (str): Path to the npy file.

    Returns:
    list: A list containing the file path and its length if the length is between 64 and 1000, inclusive.
    None: If the length is not within the valid range.
    """
    length = np.load(npy_file).shape[1]
    if 64 <= length <= 1000:
        return [npy_file, length]
    return None

def make_filelist_libritts(output_dir: str) -> None:
    """
    Create filelists for LibriTTS dataset by processing .npy files.

    Parameters:
    output_dir (str): Directory containing the preprocessed .npy files.

    Returns:
    None
    """
    output_dir = Path(output_dir)

    train_npy_files = list(output_dir.glob('train/*/*.npy'))
    test_npy_files = list(output_dir.glob('test/*/*.npy'))

    mp_func = partial(mp_npy)
    with Pool(40) as pool:
        train_results = list(tqdm(pool.imap(mp_func, train_npy_files), total=len(train_npy_files)))
        train_valid_files = [result for result in train_results if result]
        train_valid_files.sort()
    with open(Path('./filelists/filelist_libritts_train.txt'), 'w') as file:
        for npy_file, length in train_valid_files:
            file.write(f'{npy_file}|{length}\n')

    with Pool(40) as pool:
        test_results = list(tqdm(pool.imap(mp_func, test_npy_files), total=len(test_npy_files)))
        test_valid_files = [result for result in test_results if result]
        test_valid_files.sort()
        random.seed(1234)
        test_valid_files = random.sample(test_valid_files, 500)
    with open(Path('./filelists/filelist_libritts_test.txt'), 'w') as file:
        for npy_file, length in test_valid_files:
            file.write(f'{npy_file}|{length}\n')

def mp_npz(npz_file: str) -> list:
    """
    Load an npz file and check if its 'mel' length is within the valid range.

    Parameters:
    npz_file (str): Path to the npz file.

    Returns:
    list: A list containing the file path and its 'mel' length if the length is between 32 and 1000, inclusive.
    None: If the length is not within the valid range.
    """
    length = np.load(npz_file)['mel'].shape[1]
    if 32 <= length <= 1000:
        return [npz_file, length]
    return None

def make_filelist_vctk(input_dir: str, output_dir: str) -> None:
    """
    Create filelists for VCTK dataset by processing .npz files.

    Parameters:
    input_dir (str): Directory containing the raw audio files.
    output_dir (str): Directory containing the preprocessed .npz files.

    Returns:
    None
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    speakers = [spk for spk in input_dir.iterdir() if spk.is_dir()]
    test_speakers = {'p270', 'p256', 'p363', 'p241', 'p300', 'p336', 'p253', 'p266'}
    train_speakers = [spk for spk in speakers if spk.name not in test_speakers]

    train_npz_files = []
    test_npz_files = []

    for speaker in train_speakers:
        train_npz_files.extend(output_dir.glob(f'{speaker.name}/*.npz'))
    for speaker in test_speakers:
        test_npz_files.extend(output_dir.glob(f'{speaker}/*.npz'))

    mp_func = partial(mp_npz)
    with Pool(40) as pool:
        train_results = list(tqdm(pool.imap(mp_func, train_npz_files), total=len(train_npz_files)))
        train_valid_files = [result for result in train_results if result]
        train_valid_files.sort()
    with open(Path('./filelists/filelist_vctk_train.txt'), 'w') as file:
        for npz_file, length in train_valid_files:
            file.write(f'{npz_file}|{length}\n')

    with Pool(40) as pool:
        test_results = list(tqdm(pool.imap(mp_func, test_npz_files), total=len(test_npz_files)))
        test_valid_files = [result for result in test_results if result]
        test_valid_files.sort()
    with open(Path('./filelists/filelist_vctk_test.txt'), 'w') as file:
        for npz_file, length in test_valid_files:
            file.write(f'{npz_file}|{length}\n')

class AudioDataset:
    """
    A custom dataset class for loading and processing audio files.

    Attributes:
    input_dir (str): Directory containing the raw audio files.
    samplerate (int): Target sampling rate.
    save_audio (bool): Flag indicating whether to save audio for fine-tuning.
    wavs (list): List of paths to the audio files.
    """
    
    def __init__(self, input_dir: str, samplerate: int, save_audio: bool):
        self.input_dir = Path(input_dir)
        self.samplerate = samplerate
        self.save_audio = save_audio
        if save_audio:
            self.wavs = list(self.input_dir.glob('wav48_silence_trimmed/**/*_mic1.flac'))
        else:
            self.wavs = list(self.input_dir.glob('**/*.flac'))
        print('wav num: ', len(self.wavs))

    def __getitem__(self, index: int):
        """
        Load and process an audio file.

        Parameters:
        index (int): Index of the audio file in the dataset.

        Returns:
        torch.FloatTensor: Processed audio tensor.
        str: Path to save the processed data.
        """
        audio, sr = librosa.load(self.wavs[index], sr=None)
        audio, _ = librosa.effects.trim(audio, top_db=20, frame_length=2048, hop_length=300)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.samplerate)
        audio = audio / np.max(np.abs(audio)) * 0.95
        basename = self.wavs[index].stem.replace('_mic1', '')
        spk_name = basename.split('_')[0]
        if self.save_audio:
            name = Path(self.output_dir) / spk_name / basename
        else:
            split = self.wavs[index].relative_to(self.input_dir).parts[0].split('-')[0]
            name = Path(self.output_dir) / split / spk_name / basename
        return torch.FloatTensor(audio), str(name)

    def __len__(self) -> int:
        """
        Get the number of audio files in the dataset.

        Returns:
        int: Number of audio files.
        """
        return len(self.wavs)

class Collate:
    """
    A custom collate function for DataLoader.

    Methods:
    __call__(batch): Collate a batch of data.
    """

    def __call__(self, batch):
        """
        Collate a batch of data.

        Parameters:
        batch (list): A batch of data.

        Returns:
        torch.FloatTensor: Audio tensor.
        str: Path to save the processed data.
        """
        return batch[0][0], batch[0][1]

def prepare_data(rank: int, data_loader: DataLoader, args: argparse.Namespace) -> None:
    """
    Prepare data for training by processing audio files and generating mel spectrograms.

    Parameters:
    rank (int): Rank of the current process.
    data_loader (DataLoader): DataLoader for the dataset.
    args (argparse.Namespace): Command line arguments.

    Returns:
    None
    """
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            audio, name = batch
            mel = mel_spectrogram_torch(audio.cuda(rank, non_blocking=True).unsqueeze(0),
                                        n_fft=2048,
                                        num_mels=128,
                                        sampling_rate=24000,
                                        hop_size=300,
                                        win_size=1200,
                                        fmin=0, fmax=None).squeeze(0)

            name_path = Path(name)
            name_path.parent.mkdir(parents=True, exist_ok=True)
            if args.save_audio:
                data = {'mel': mel.cpu(), 'audio': audio.cpu()}
                np.savez(name_path, **data)
            else:
                np.save(name_path, mel.cpu())

def run(rank: int, n_gpus: int, args: argparse.Namespace) -> None:
    """
    Run the training process on a specific GPU.

    Parameters:
    rank (int): Rank of the current process.
    n_gpus (int): Number of GPUs available.
    args (argparse.Namespace): Command line arguments.

    Returns:
    None
    """
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(1234)
    torch.cuda.set_device(rank)

    dataset = AudioDataset(args.input_dir, args.samplerate, args.save_audio)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=n_gpus, rank=rank, shuffle=True)
    collate_fn = Collate()
    data_loader = DataLoader(dataset, num_workers=16, shuffle=False,
                             batch_size=1, pin_memory=True,
                             drop_last=False, collate_fn=collate_fn, sampler=sampler)

    prepare_data(rank, data_loader, args)

def main(args: argparse.Namespace) -> None:
    """
    Main function to initialize the training process.

    Parameters:
    args (argparse.Namespace): Command line arguments.

    Returns:
    None
    """
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    port = 60000 + rdint(0, 1000)
    os.environ['MASTER_PORT'] = str(port)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args))
    
    filelists_dir = Path('./filelists')
    filelists_dir.mkdir(exist_ok=True)
    
    if args.save_audio:
        make_filelist_vctk(args.input_dir, args.output_dir)
    else:
        make_filelist_libritts(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='/data1/DB/VCTK/VCTK/data/VCTK/wav48_silence_trimmed', help='Directory of audio files')
    parser.add_argument('-o', '--output_dir', default='/data1/projects/frepainter/preprocessed_data/VCTK', help='Directory to save preprocessed data')
    parser.add_argument('-s', '--samplerate', default=24000, help='Target sampling rate')
    parser.add_argument('--save_audio', action='store_true', help='Saving audio for fine-tuning (True: VCTK, False: LibriTTS)')
    args = parser.parse_args()

    main(args)
