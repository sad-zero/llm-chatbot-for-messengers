# LLM Chatbot for messengers
Time bound chatbot for various messengers.

---

## Overview
This project aims to solve the time-bound issue that arises when integrating LLM-based chatbots with messenger platforms.<br/>
While LLM chatbots enable more natural and conversational interactions, they often require more time to generate responses for complex scenarios.<br/>
Some messengers enforce strict response time limits for third-party chatbots, and if these limits are exceeded, responses are forcibly rejected.<br/>
As a result, users may not receive chatbot replies, even if the chatbot generates them successfully. <br/>
This project addresses this challenge by providing a solution that ensures timely responses, enabling smoother integration of LLM chatbots with messenger platforms.<br/>
By using this project, developers can create LLM-based chatbots that comply with messenger time constraints more effectively.

## Main Senario
```mermaid
flowchart TD
    User --Ask a question--> Chatbot
    Chatbot --Generate an answer--> LLM
    Chatbot --Check timeout--> TimeLimitChecker
    LLM & TimeLimitChecker --> timeout{is time over?}
    timeout --Y--> FallbackMessage
    timeout --N--> GeneratedAnswer
```
