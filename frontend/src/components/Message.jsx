import ReactMarkdown from "react-markdown";
import { Sparkles, User } from "lucide-react";

function Message({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`message-avatar ${isUser ? "user-avatar" : "ai-avatar"}`}>
        {isUser ? (
          <User size={17} />
        ) : (
          <Sparkles size={17} />
        )}
      </div>

      <div className={`message-content ${isUser ? "user-content" : ""}`}>
        <div className="message-name">
          {isUser ? "You" : "CA Mentor AI"}
        </div>

        <div className="message-text">
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown>{message.content}</ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  );
}

export default Message;