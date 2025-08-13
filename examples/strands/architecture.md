flowchart LR
  U[User / Prompt]
  SA[StrandsAgent\nConstruct message & turn history\nRecord Trajectory]
  RM[RLLMModel.stream\nConvert to Chat message\nCall RolloutEngine\nRecord I/O]
  RE[RolloutEngine\nSend to backend model]
  LM[LLM Qwen or 4o\nReturn plain text]
  PARSE[parse_action & normalize\nValidate against action whitelist\nFill in or rewrite parameters]
  BRW[Browser Tool local call]
  OBS[Observation\nText or structured result]
  LOG[Trajectory record]

  U --> SA --> RM --> RE --> LM --> RM
  RM --> SA
  SA --> PARSE --> BRW --> OBS --> SA
  SA --> LOG
