import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }


def evaluate_trainer(trainer, test_dataset, model_name_str):
    results = trainer.evaluate(test_dataset)
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels_true = predictions_output.label_ids

    report = classification_report(labels_true, preds, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)

    print(f"\nРезультаты модели: {model_name_str}")
    print(f"   Accuracy:     {results['eval_accuracy']:.4f}")
    print(f"   F1 Macro:     {results['eval_f1_macro']:.4f}")

    return {
        "name": model_name_str,
        "accuracy": results['eval_accuracy'],
        "f1_macro": results['eval_f1_macro'],
        "report": report
    }