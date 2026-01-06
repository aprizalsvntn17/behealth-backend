import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.trainer import train_model


def main():
    print("=" * 70)
    print(" " * 15 + "HEALTH CHATBOT - MODEL TRAINING")
    print("=" * 70)
    print()
    
    if not os.path.exists(Config.DATASET_PATH):
        print(f"ERROR: Dataset not found at {Config.DATASET_PATH}")
        sys.exit(1)
    
    print(f"Dataset: {Config.DATASET_PATH}")
    print(f"Model output: {Config.MODEL_PATH}")
    print(f"Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
    print()
    
    best_classifier = 'logistic'
    
    print(f"Starting training with {best_classifier.upper()} classifier...")
    print()
    
    try:
        result = train_model(
            dataset_path=Config.DATASET_PATH,
            model_path=Config.MODEL_PATH,
            classifier_type=best_classifier
        )
        
        print("\n" + "=" * 70)
        print(" " * 25 + "TRAINING SUCCESS")
        print("=" * 70)
        print()
        print(f"Final Results:")
        print(f"   Classifier: {best_classifier.upper()}")
        print(f"   Test Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"   CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
        print(f"   Number of Intents: {result['num_classes']}")
        print(f"   Training Samples: {result['num_samples']}")
        print()
        print(f"Models saved to: {Config.MODEL_PATH}")
        print()
        print("You can now start the server with:")
        print("   python app.py")
        print()
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(" " * 28 + "TRAINING FAILED")
        print("=" * 70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()