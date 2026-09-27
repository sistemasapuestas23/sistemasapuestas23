import argparse
from datetime import datetime, timedelta
from pipeline.full_pipeline import FullPipeline
from utils.logging_config import setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Football Prediction Engine — TheStatsAPI"
    )
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument(
        "--competition_id",
        required=True,
        help="TheStatsAPI competition_id e.g. comp_3039",
    )
    parser.add_argument(
        "--season_ids", nargs="+", help="List of season_ids for training"
    )
    parser.add_argument("--date_from", help="YYYY-MM-DD for prediction window")
    parser.add_argument("--date_to", help="YYYY-MM-DD for prediction window")
    args = parser.parse_args()

    pipeline = FullPipeline()
    try:
        if args.mode == "train":
            if not args.season_ids:
                raise ValueError("--season_ids required for training")
            metrics = pipeline.run_training(args.competition_id, args.season_ids)
            print("Training complete. OOF metrics:", metrics)
        else:
            date_from = args.date_from or datetime.utcnow().strftime("%Y-%m-%d")
            date_to = args.date_to or (datetime.utcnow() + timedelta(days=7)).strftime(
                "%Y-%m-%d"
            )
            preds = pipeline.predict_upcoming(args.competition_id, date_from, date_to)
            print(preds.to_string(index=False))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
