import pathlib

from langchain.chat_loaders import whatsapp


def test_whatsapp_chat_loader() -> None:
    chat_path = pathlib.Path(__file__).parent / "data" / "whatsapp_chat.txt"
    loader = whatsapp.WhatsAppChatLoader(str(chat_path), user_name="Dr. Feather")

    chat_sessions = loader.load()
    assert chat_sessions, "Chat sessions should not be empty"

    assert chat_sessions[0]["messages"], "Chat messages should not be empty"

    assert (
        "I spotted a rare Hyacinth Macaw yesterday in the Amazon Rainforest."
        " Such a magnificent creature!" in chat_sessions[0]["messages"][0].content
    ), "Chat content mismatch"
