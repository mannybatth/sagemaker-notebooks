import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from datasets import load_from_disk, load_metric
from transformers import LayoutLMv2ForTokenClassification, AutoProcessor, LayoutLMv2Tokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str,
                        default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str,
                        default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str,
                        default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str,
                        default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str,
                        default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(
        f" torch version: {torch.__version__} cuda: {torch.cuda.is_available()}")

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # Prepare model labels - useful in inference API
    labels_list = train_dataset.features["labels"].feature.names
    num_labels = len(labels_list)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels_list):
        label2id[label] = i
        id2label[i] = label

    metric = load_metric("seqeval")

    return_entity_level_metrics = False

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
#         predictions = np.argmax(predictions, axis=1)

        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(
            predictions=true_predictions, references=true_labels, zero_division='0')
        if return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # download model from model hub
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
#     tokenizer = AutoProcessor.from_pretrained(args.model_id, apply_ocr=False)
    tokenizer = LayoutLMv2Tokenizer.from_pretrained(args.model_id)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True if get_last_checkpoint(
            args.output_dir) is not None else False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        logging_strategy="epoch",
        evaluation_strategy="no",
        save_strategy="no",
        save_total_limit=1,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    logger.info("***** create Trainer instance *****")

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    logger.info("***** start train *****")

    # train model
    if get_last_checkpoint(args.output_dir) is not None:
        logger.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    logger.info("***** start evaluate *****")

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    logger.info("***** generate results *****")

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            print(f"{key} = {value}\n")

    logger.info("***** save model *****")

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
