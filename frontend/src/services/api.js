const API_BASE_URL = "http://127.0.0.1:8000";

export async function sendMessage(question) {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
    }),
  });

  if (!response.ok) {
    let message = "Something went wrong while processing your question.";

    try {
      const data = await response.json();

      if (data?.detail) {
        message =
          typeof data.detail === "string"
            ? data.detail
            : "The server could not process your question.";
      }
    } catch {
      // Keep default error message.
    }

    throw new Error(message);
  }

  return response.json();
}