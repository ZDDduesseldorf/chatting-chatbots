import pytest
from chatbots.chatbot_eliza_style.eliza_like_bot import eliza_answer


@pytest.mark.parametrize('user_input, expected_outputs', [
    ["I need my dog.", ["Would it really help you to get your dog?",
                        "Are you sure you need your dog?",
                        "Why do you need your dog?"]],
    ["Yes", ["You seem quite sure.",
             "OK, but can you elaborate a bit?"]],
    ["asldkjhalkjhd d kjhasdkjh?", ["Why do you ask that?",
                                    "Please consider whether you can answer your own question.",
                                    "Perhaps the answer lies within yourself?",
                                    "Why don't you tell me?"]],
])
def test_eliza_conversations(user_input, expected_outputs):
    eliza_output = eliza_answer(user_input)
    assert eliza_output in expected_outputs, "Unexpected chatbot answer!"
