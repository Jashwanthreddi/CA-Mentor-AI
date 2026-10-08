import {
  BookOpen,
  Calculator,
  FileText,
  Scale,
} from "lucide-react";

const suggestions = [
  {
    icon: BookOpen,
    title: "Accounting",
    question: "What is accounting?",
  },
  {
    icon: Scale,
    title: "Law",
    question: "Explain the meaning of consideration in contract law.",
  },
  {
    icon: FileText,
    title: "Business",
    question: "What are the important functions of management?",
  },
  {
    icon: Calculator,
    title: "Quantitative Aptitude",
    question: "Explain the arithmetic mean with an example.",
  },
];

function Welcome({ onSuggestion }) {
  return (
    <div className="welcome">
      <div className="welcome-icon">
        <span>✦</span>
      </div>

      <h1>Welcome to CA Mentor AI</h1>

      <p>
        Your intelligent study companion for CA preparation.
        <br />
        Ask questions, understand concepts, and learn with confidence.
      </p>

      <div className="suggestions">
        {suggestions.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.title}
              className="suggestion-card"
              onClick={() => onSuggestion(item.question)}
            >
              <Icon size={18} strokeWidth={1.8} />

              <span>{item.title}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default Welcome;