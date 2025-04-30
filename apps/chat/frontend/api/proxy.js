// api/proxy.js
export default async (req, res) => {
  // Set CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  // Handle preflight request
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  // Only allow POST requests
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    // Create a fetch request with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout

    // Forward the request to your Google Compute Engine API
    const response = await fetch("http://34.47.155.223:8000/rag/answer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req.body),
      signal: controller.signal,
    });

    // Clear the timeout
    clearTimeout(timeoutId);

    // Get response data
    const data = await response.json();

    // Return the API response
    return res.status(response.status).json(data);
  } catch (error) {
    console.error("Proxy error:", error);

    // Check if it's a timeout error
    if (error.name === "AbortError") {
      return res.status(504).json({
        error: "Request to external API timed out",
        message:
          "The server at 34.47.155.223:8000 is taking too long to respond",
      });
    }

    return res.status(500).json({
      error: "Failed to reach API server",
      message: error.message,
    });
  }
};
