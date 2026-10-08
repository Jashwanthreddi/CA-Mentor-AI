import { useState } from "react";
import { Plus, Sparkles } from "lucide-react";

import Welcome from "./components/Welcome";
import Message from "./components/Message";
import ChatInput from "./components/ChatInput";
import { sendMessage } from "./services/api";

function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const hasMessages = messages.length > 0;

  const handleSend = async (question) => {
    if (loading) {
      return;
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
    ]);

    setLoading(true);

    try {
      const data = await sendMessage(question);

      const answer =
        data?.answer ??
        data?.data?.answer ??
        data?.response ??
        "I couldn't generate an answer for that question.";

      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: answer,
      };

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);
    } catch (error) {
      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          error.message ||
          "Sorry, something went wrong. Please try again.",
        isError: true,
      };

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    if (loading) {
      return;
    }

    setMessages([]);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={17} />
          </div>

          <div>
            <div className="brand-name">CA Mentor AI</div>
            <div className="brand-subtitle">
              Your CA study companion
            </div>
          </div>
        </div>

        <button
          className="new-chat-button"
          onClick={handleNewChat}
          disabled={loading}
        >
          <Plus size={17} />
          <span>New chat</span>
        </button>
      </header>

      <main className={`chat-area ${hasMessages ? "has-messages" : ""}`}>
        {!hasMessages ? (
          <Welcome onSuggestion={handleSend} />
        ) : (
          <div className="messages-container">
            {messages.map((message) => (
              <Message
                key={message.id}
                message={message}
              />
            ))}

            {loading && (
              <div className="message-row assistant-row">
                <div className="message-avatar ai-avatar">
                  <Sparkles size={17} />
                </div>

                <div className="message-content">
                  <div className="message-name">
                    CA Mentor AI
                  </div>

                  <div className="typing-indicator">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="composer-container">
          <ChatInput
            onSend={handleSend}
            disabled={loading}
          />

          <div className="composer-note">
            CA Mentor AI can make mistakes. Verify important information
            with your study material.
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;