import pytest
from unittest.mock import patch, MagicMock
from nutrition_detector.training.trainer import train
import argparse
from tests.mocks import MockProcessor, MockModel

@patch("nutrition_detector.training.trainer.get_model_and_processor")
@patch("nutrition_detector.training.trainer.SFTTrainer")
@patch("nutrition_detector.training.trainer.load_dataset")
def test_train_dry_run(mock_load, mock_trainer_cls, mock_get_model):
    # Setup mocks
    mock_model = MockModel()
    mock_processor = MockProcessor()
    
    mock_get_model.return_value = (mock_model, mock_processor)
    
    args = argparse.Namespace(
        model_id="mock-model",
        dataset_id="mock-data",
        output_dir="mock-output",
        epochs=1,
        batch_size=1,
        max_samples=None,
        dry_run=True
    )
    
    train(args)
    
    # Assert get_model_and_processor called
    mock_get_model.assert_called_once_with(model_id="mock-model", dry_run=True)
    
    # Assert SFTTrainer initialized (even in dry run we init it to check config)
    assert mock_trainer_cls.called
    
    # Assert train NOT called on trainer instance in dry_run
    trainer_instance = mock_trainer_cls.return_value
    trainer_instance.train.assert_not_called()

@patch("nutrition_detector.training.trainer.get_model_and_processor")
@patch("nutrition_detector.training.trainer.SFTTrainer")
@patch("nutrition_detector.training.trainer.load_dataset")
def test_train_real_run_logic(mock_load, mock_trainer_cls, mock_get_model):
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_get_model.return_value = (mock_model, mock_processor)
    
    args = argparse.Namespace(
        model_id="mock-model",
        dataset_id="mock-data",
        output_dir="mock-output",
        epochs=1,
        batch_size=1,
        max_samples=None,
        dry_run=False
    )
    
    train(args)
    
    trainer_instance = mock_trainer_cls.return_value
    trainer_instance.train.assert_called_once()
    trainer_instance.save_model.assert_called_once_with("mock-output")
