import json
import os
import argparse

SUBSETS = ['train', 'test', 'val']


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='json annotation file to be used')
    parser.add_argument('directory', help='path to folder rawframes')
    parser.add_argument('--video_dataset', '-vid', action='store_true', default=False, help='create video dataset annotation')
    return parser.parse_args()


def delete_existing_annotations():
    # Delete existing annotation files
    for subset in SUBSETS:
        try:
            os.remove(f'{subset}_rawframes_annotations.txt')
        except:
            pass

    for subset in SUBSETS:
        try:
            os.remove(f'{subset}_videodataset_annotations.txt')
        except:
            pass


def load_annotations(args):
    # Load the json labels
    with open(args.json_file) as fin:
        videos = json.load(fin)
    return videos


def write_annotations(videos, video_annotations):
    # Create the annotation files
    for video_id in videos:
        class_id = videos[video_id]['action'][0]
        subset = videos[video_id]['subset']
        directory = f'rawframes/{subset}/{video_id}'

        # Create RawFrames annotations
        frames = len([frame for frame in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, frame))])
        with open(f'{subset}_annotations.txt', 'a') as fout:
            fout.write(f'{subset}/{video_id} {frames} {class_id}\n')

        if video_annotations:
            # Create VideoDataset annotations
            with open(f'{subset}_videodataset_annotations.txt', 'a') as fout:
                fout.write(f'{subset}/{video_id}.mp4 {class_id}\n')


if __name__ == '__main__':
    args = load_args()
    videos = load_annotations(args)
    os.chdir(args.directory)
    delete_existing_annotations()
    write_annotations(videos, args.video_dataset)
