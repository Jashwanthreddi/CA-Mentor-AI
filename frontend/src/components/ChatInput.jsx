import { ArrowUp } from "lucide-react";
import { useState } from "react";

function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Ask anything about your CA studies...",
}) {
  const [value, setValue] = useState("");

  const submit = () => {
    const question = value.trim();

    if (!question || disabled) {
      return;
    }

    onSend(question);
    setValue("");
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="input-wrapper">
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        rows={1}
        disabled={disabled}
      />

      <button
        className="send-button"
        onClick={submit}
        disabled={!value.trim() || disabled}
        aria-label="Send message"
      >
        <ArrowUp size={19} />
      </button>
    </div>
  );
}

export default ChatInput;