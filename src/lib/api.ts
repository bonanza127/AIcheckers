const getApiUrl = () => {
  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;
    if (hostname === "aicheckers.net" || hostname === "www.aicheckers.net" || hostname.endsWith(".vercel.app")) {
      return "https://api.aicheckers.net";
    }
  }
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
};

export default getApiUrl;
