# convert_to_tflite.py

import tensorflow as tf
import os
from pathlib import Path
import argparse


def convert_model_to_tflite(model_path: str, tflite_model_path: str, quantize: bool = False):
    """Convert a Keras model to TFLite.

    Args:
        model_path: path to Keras model (.h5 or saved_model).
        tflite_model_path: output .tflite path.
        quantize: apply post-training dynamic range quantization when True.
    """
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        # Fallback: try allowing Select TF ops (larger binary) and disable
        # experimental lowering of tensor list ops which can fail for some RNNs.
        print("Standard TFLite conversion failed with error:", e)
        print("Retrying with Select TF ops and disabling tensor-list lowering (may increase binary size and require TF Select runtime).")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        # This flag name is internal/experimental; still commonly used as a workaround.
        try:
            converter._experimental_lower_tensor_list_ops = False
        except Exception:
            pass
        tflite_model = converter.convert()

    out_dir = Path(tflite_model_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to: {tflite_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='model/saved/neuro_model.h5')
    parser.add_argument('--output', type=str, default='model/tflite/neuro_model.tflite')
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()

    convert_model_to_tflite(args.model_path, args.output, quantize=args.quantize)