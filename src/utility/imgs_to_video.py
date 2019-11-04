#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Action distribution mapping.
# =============================================================================
import argparse

from utils import images_to_video, parse_path

def main():
    parser = argparse.ArgumentParser(description="yumi_push_act_dist")
    parser.add_argument("--out", type=str, default="",
                        help="outs directory that should be evaluated")
    parser.add_argument("--policy", type=str, default="",
                        help="policy directory that should be evaluated")
    parser.add_argument("--tag", type=str, default="",
                        help="image names to look for while creating video")
    parser.add_argument("--video_name", type=str, default="")
    args = parser.parse_args()
    if not ".avi" in args.video_name: args.video_name = args.video_name + ".avi"

    directory, path = parse_path(args)
    images_to_video(path, args.tag, args.video_name)

if __name__ == "__main__":
    main()
