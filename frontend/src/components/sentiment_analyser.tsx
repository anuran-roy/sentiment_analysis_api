import React, { useState, useEffect } from "react";

export default function SentimentAnalyzer() {
  const [text, setText] = useState<any>("Hello!");
  const [sentiment, setSentiment] = useState<any>("None");

  const getSentiment = () => {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence: text }),
    };
    fetch(`http://localhost:8000/app/inference`, requestOptions)
      .then((response) => response.json())
      .then((data) => setSentiment(data.sentiment));
  };
  // useEffect(() => { getSentiment() }, [text])
  return (
    <div className="flex h-screen w-screen flex-col items-center justify-center object-center text-center">
      <textarea
        className="border-2 shadow-lg"
        placeholder={text}
        value={text}
        onChange={(e: any) => setText(e.target.value)}
      />
      <button onClick={getSentiment} className="px-5 py-3 mx-3 my-1 border-2 bg-green-700">Get Sentiment</button>
      <div className="p-5 text-4xl font-bold text-gray-700">{sentiment}</div>
    </div>
  );
}
