import React from "react";
import logo from "./logo.svg";
import "./App.css";
import SentimentAnalyzer from "./components/sentiment_analyser";

function App() {
  return (
    <div className="App">
      <div className="text-4xl font-bold italic">Sentiment Analyzer App</div>
      <SentimentAnalyzer></SentimentAnalyzer>
    </div>
  );
}

export default App;
