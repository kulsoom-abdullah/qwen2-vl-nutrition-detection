import pytest
from unittest.mock import MagicMock
from nutrition_detector.data.dataset import parse_bounding_boxes, create_chat_format, SYSTEM_MESSAGE, USER_PROMPT

def test_parse_bounding_boxes_standard():
    response = "nutrition-table<box(100,100),(200,200)>"
    # 100/1000 = 0.1
    expected = [[0.1, 0.1, 0.2, 0.2]]
    assert parse_bounding_boxes(response) == expected

def test_parse_bounding_boxes_multiple():
    response = "nutrition-table<box(100,100),(200,200)>\nnutrition-table<box(300,300),(400,400)>"
    expected = [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]
    assert parse_bounding_boxes(response) == expected

def test_parse_bounding_boxes_floats():
    response = "box(100.5, 100.5), (200.5, 200.5)"
    # 100.5/1000 = 0.1005
    expected = [[0.1005, 0.1005, 0.2005, 0.2005]]
    assert parse_bounding_boxes(response) == expected

def test_parse_bounding_boxes_empty():
    assert parse_bounding_boxes("no boxes here") == []

def test_create_chat_format():
    mock_image = MagicMock()
    mock_image.copy.return_value = mock_image
    
    sample = {
        "image": mock_image,
        "objects": {
            "bbox": [[0.1, 0.2, 0.3, 0.4]], # y_min, x_min, y_max, x_max (dataset format)
            "category_name": ["table"]
        }
    }
    
    # 0.2 * 1000 = 200 (x_min)
    # 0.1 * 1000 = 100 (y_min)
    # 0.4 * 1000 = 400 (x_max)
    # 0.3 * 1000 = 300 (y_max)
    
    result = create_chat_format(sample, downsize=False)
    
    messages = result["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    
    # Check assistant response format
    # x_min, y_min, x_max, y_max
    expected_box_str = "(200,100),(400,300)"
    assert expected_box_str in messages[2]["content"]
    assert "<|object_ref_start|>table<|object_ref_end|>" in messages[2]["content"]
