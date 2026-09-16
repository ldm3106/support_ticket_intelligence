import pandas as pd
import pytest
from support_ticket.preprocessing import prepare_text, text_key
from support_ticket.data import split_data


def test_preprocessing_retains_meaning():
    assert prepare_text("  I can't\n log in!  ") == "I can't log in!"
    assert text_key(" CARD\tProblem ") == "card problem"


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_text_rejected(text):
    with pytest.raises(ValueError):
        prepare_text(text)


def test_non_string_rejected():
    with pytest.raises(TypeError):
        prepare_text(None)


def test_splits_are_disjoint_and_label_ids_stable():
    train = pd.DataFrame([{"text": f"{label} ticket {i}", "label": label}
                          for label in ("a", "b") for i in range(20)])
    train = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    test = pd.DataFrame([{"text": "A ticket 0", "label": "a"}, {"text": "new b", "label": "b"}])
    frames, labels = split_data(train, test)
    assert labels == ["a", "b"]
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(frames[first].key) & set(frames[second].key)
    assert sum(len(frames[k]) for k in ("train", "validation")) == 39
    assert frames["test"].label_id.tolist() == [0, 1]


def test_unknown_test_label_rejected():
    with pytest.raises(ValueError, match="unknown labels"):
        split_data(pd.DataFrame({"text": ["hello"], "label": ["a"]}),
                   pd.DataFrame({"text": ["bye"], "label": ["b"]}))


def test_conflicting_training_text_is_removed():
    train = pd.DataFrame([{"text": f"{label} {i}", "label": label}
                          for label in ("a", "b") for i in range(20)] +
                         [{"text": "ambiguous", "label": "a"}, {"text": "AMBIGUOUS", "label": "b"}])
    test = pd.DataFrame({"text": ["new a", "new b"], "label": ["a", "b"]})
    frames, _ = split_data(train, test)
    assert all("ambiguous" not in set(frames[name].key) for name in ("train", "validation"))


def test_saved_model_tokenization_and_inference():
    from support_ticket.config import MODELS
    from support_ticket.inference import TicketPredictor
    if not (MODELS / "transformer" / "config.json").exists():
        pytest.skip("Run training to enable saved-model integration check")
    predictor = TicketPredictor()
    encoded = predictor.tokenizer(["Help", "I cannot use my card " * 100], padding=True,
                                  truncation=True, max_length=predictor.max_length, return_tensors="pt")
    assert encoded["input_ids"].shape == encoded["attention_mask"].shape
    assert encoded["input_ids"].shape[1] <= predictor.max_length
    assert 0 in encoded["attention_mask"][0]
    result = predictor.predict_ticket("I was charged twice for a card payment")
    assert result["predicted_category"] in predictor.model.config.label2id
    assert 0 <= result["confidence_score"] <= 1
    assert not result["truncated"]
    assert result == predictor.predict_ticket("I was charged twice for a card payment")
    assert predictor.predict_ticket("card " * 150)["truncated"]
    from support_ticket.train import tokenize_frame
    dataset = tokenize_frame(pd.DataFrame({"text": ["Help with my card"], "label_id": [3]}),
                             predictor.tokenizer, predictor.max_length)
    assert dataset[0]["labels"] == 3
    assert len(dataset[0]["input_ids"]) == len(dataset[0]["attention_mask"])
