"""
Traffic CCTV Analysis - Main Entry Point

Usage:
    python main.py run --video Road_2.mp4              # Single stream
    python main.py run --multi                          # Multi-stream (auto-detect)
    python main.py setup --video Road_2.mp4 --mode all  # Setup zones
"""

import argparse
from pathlib import Path


def run_single(args):
    from stream_processor import StreamProcessor
    
    video = args.video
    if not Path(video).exists():
        video = f"Option1/{args.video}"
    
    config = args.config or f"configs/{Path(video).stem}.json"
    
    if not Path(config).exists():
        print(f"Config not found: {config}")
        print("Run: python main.py setup --video YOUR_VIDEO.mp4 --mode all")
        return
    
    p = StreamProcessor(video, config, args.model, display=not args.no_display, save_video=not args.no_save)
    p.process()


def run_multi(args):
    from multi_stream import MultiStreamManager, StreamConfig
    
    streams = []
    for i in range(1, 5):  # Road_1 đến Road_4
        v = Path(f"Option1/Road_{i}.mp4")
        c = Path(f"configs/Road_{i}.json")
        if v.exists() and c.exists():
            streams.append(StreamConfig(str(v), str(c)))
    
    if not streams:
        print("No configured streams. Run setup_zones.py for each video first.")
        return
    
    mgr = MultiStreamManager(streams, args.model)
    mgr.start()


def run_setup(args):
    import subprocess
    cmd = ["python", "setup_zones.py", "--video", args.video, "--mode", args.mode]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Traffic CCTV Analysis")
    sub = parser.add_subparsers(dest='cmd')
    
    # Setup
    setup = sub.add_parser('setup', help='Setup zones')
    setup.add_argument('--video', '-v', required=True)
    setup.add_argument('--mode', '-m', default='all', choices=['perspective', 'roi', 'lane-divider', 'all'])
    
    # Run
    run = sub.add_parser('run', help='Run analysis')
    run.add_argument('--video', '-v', help='Video for single stream')
    run.add_argument('--config', '-c', help='Config path (auto-detect if not set)')
    run.add_argument('--multi', action='store_true', help='Multi-stream mode')
    run.add_argument('--model', '-m', default='yolo11n.pt')
    run.add_argument('--no-display', action='store_true')
    run.add_argument('--no-save', action='store_true')
    
    args = parser.parse_args()
    
    if args.cmd == 'setup':
        run_setup(args)
    elif args.cmd == 'run':
        if args.multi:
            run_multi(args)
        elif args.video:
            run_single(args)
        else:
            print("Specify --video or --multi")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
