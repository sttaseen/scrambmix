import json
import os
import argparse

def load_args():
    """
    Load command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='path to the uncompressed archive')
    parser.add_argument('-n', '--samples', type=int, default=None, help='number of samples per class for a subset')
    return parser.parse_args()

def move_test_videos():
    """
    Move videos from 'videos' to 'test' folder based on information in 'test_labels.json'.

    Raises:
        FileNotFoundError: If a video file is not found.
    """
    os.makedirs('test', exist_ok=True)
    with open('test_labels.json') as fin:
        videos = json.load(fin)

    # Move the videos into respective folders
    for video in videos:
        try:
            video_id = video['id']
            os.rename(f'videos/{video_id}.webm', f'test/{video_id}.webm')
        except FileNotFoundError:
            print(f'{video_id}.webm not found. Skipping and going next...')

def move_val_videos():
    """
    Move videos from 'videos' to 'val' folder based on information in 'val_labels.json'.

    Raises:
        FileNotFoundError: If a video file is not found.
    """
    os.makedirs('val', exist_ok=True)
    with open('val_labels.json') as fin:
        videos = json.load(fin)

    # Move the videos into respective folders
    for video in videos:
        try:
            video_id = video['id']
            os.rename(f'videos/{video_id}.webm', f'val/{video_id}.webm')
        except FileNotFoundError:
            print(f'{video_id}.webm not found. Skipping and going next...')

def move_train_videos(max_samples):
    """
    Move videos from 'videos' to 'train' folder based on information in 'train_labels.json',
    considering the maximum samples per class.

    Args:
        max_samples (int): Maximum samples per class.

    Raises:
        FileNotFoundError: If a video file is not found.
        AssertionError: If max_samples exceeds the limit of 91.
    """
    if max_samples is not None:
        assert max_samples <= 91, "min samples per class is 91"
        count = dict()
    
    os.makedirs('train', exist_ok=True)
    with open('train_labels.json') as fin:
        videos = json.load(fin)

    # Move the videos into respective folders
    for video in videos:
        try:
            video_id = video['id']
            class_id = video['class']
            
            if max_samples is None:
                os.rename(f'videos/{video_id}.webm', f'train/{video_id}.webm')
                continue

            # Creating a subset
            if class_id not in count:
                count[class_id] = 1
            elif count[class_id] < max_samples:
                os.rename(f'videos/{video_id}.webm', f'train/{video_id}.webm')
                count[class_id] = count[class_id] + 1
        except FileNotFoundError:
            print(f'{video_id}.webm not found. Skipping and going next...')

if __name__ == '__main__':
    args = load_args()
    os.chdir(args.directory)
    move_test_videos()
    move_val_videos()
    move_train_videos(max_samples=args.samples)
