"""Upload a recorded mp4 to a W&B run (resumes the training run by id so the video
appears alongside the training curves)."""
import argparse
import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--entity", default="tashapais")
    ap.add_argument("--project", default="epymarl-smacv2-mappo")
    ap.add_argument("--run-id", required=True, help="existing run id to resume (e.g. c0tpvzrc)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--key", default="trained_agents_video")
    args = ap.parse_args()

    run = wandb.init(entity=args.entity, project=args.project, id=args.run_id, resume="allow")
    run.log({args.key: wandb.Video(args.video, fps=args.fps, format="mp4")})
    run.finish()
    print(f"Uploaded {args.video} to wandb run {args.entity}/{args.project}/{args.run_id}")


if __name__ == "__main__":
    main()
