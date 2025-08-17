# Qwen Model Debugging Log

## Problem Statement
Qwen model (Qwen/Qwen2.5-7B-Instruct-Turbo) getting stuck in repetitive loops with JSON parsing errors when running strands_rllm_browser_agent_v2.py.

## Root Causes Identified

### 1. JSON Parsing Failures
- **Issue**: Model generates malformed JSON with missing closing braces
- **Example**: `Action: {"name":"browser","arguments":{"action":{"type":"navigate"...}}[step 1]`
- **Impact**: Parse failures cause repeated attempts

### 2. Repetitive Loop Behavior  
- **Issue**: Model doesn't learn from previous actions/results
- **Pattern**: Steps 1-6 same navigate, Steps 8-17 same get_text
- **Root**: No memory/state tracking between actions

### 3. Token Context Explosion
- **Issue**: Context grows unbounded until hitting 32769 limit
- **Trigger**: `36989 + 350 > 32769` tokens
- **Result**: Complete failure with 422 errors

### 4. Memory/State Issues
- **Issue**: Model forgets what it just did
- **Impact**: Doesn't build on previous observations

## Solution Plan (Hybrid Approach A+C+D)

### ✅ COMPLETED
- [x] **JSON Parsing Robustness**: Enhanced parse_action() with malformed JSON repair
- [x] **Loop Detection**: Added ActionHistory class to track repetitive actions  
- [x] **Context Management**: Implemented token estimation and history truncation
- [x] **Better Error Handling**: Added parse error limits and suggestions

### 🔄 IN PROGRESS  
- [ ] **Async Race Conditions**: Review async/await patterns
- [ ] **Fallback Strategies**: Add graceful degradation

### ⏳ PENDING
- [ ] **Model Parameter Tuning**: Test with lower temperature (0.1-0.3)
- [ ] **Qwen-Specific Prompts**: Optimize for Qwen behavior patterns

## Implementation Status

### Changes Made to strands_rllm_browser_agent_v2.py:

1. **Enhanced JSON Parsing** (Line 74-118)
   - Fixed regex pattern for better JSON extraction
   - Added malformed JSON repair logic
   - Handles missing braces and incomplete JSON

2. **ActionHistory Class** (Line 120-155)
   - Tracks last 10 actions with repetition counting
   - Detects loops (3+ consecutive identical actions)
   - Provides smart suggestions for alternatives

3. **Loop Prevention in BrowserExecutor** (Line 193-217)
   - Integrates ActionHistory into execution
   - Returns structured errors for repetitive actions
   - Suggests alternative approaches

4. **Improved System Prompt** (Line 18-47)
   - Clearer workflow instructions
   - Emphasizes no repetition rules
   - Better error handling guidance

5. **Context Management** (Line 320-340)
   - Reduced max_steps from 50 to 25
   - Token estimation and truncation
   - Smart history management (keep last 6 interactions)

6. **Better Error Recovery** (Line 419-435)
   - Parse error limits (max 3 before forcing conclusion)
   - Fallback conclusion generation
   - More actionable error messages

## Test Results

### ✅ Progress Made
- Tool_calls parsing works properly 
- No more JSON malformed errors
- Basic navigation working

### ❌ Issues Remaining
1. **Missing Context Display**: Only showing tool_calls, no LLM reasoning/thoughts
2. **Hard-coded Loop Detection Unsustainable**: Model hits repetitive_action_detected but doesn't learn to change behavior
3. **Model Still Loops**: Steps 5-6 show same navigate action despite error messages

### Root Issue Identified ⚠️ **CRITICAL**
We're **bypassing Strands' built-in memory system**! 

**Strands Agent 已经有 `self.messages`** (Line 367 in strands.py):
- 每次调用 `invoke_async()` 都会自动管理对话历史
- 我们不需要手动维护 `history: list[str]` 
- **StrandsAgent.chat_completions** 已经提供了完整的对话上下文

**我们的错误**：
- 手动构建 `chat_messages = [system, user]` (Line 335-338) 
- 完全忽略了 Strands 的 `self.messages` 
- 重复造轮子而不是用框架功能

**正确做法**：
1. **我们不应该管记忆** - 那是 Strands 的责任
2. **我们只负责 RL tracking** - Step/Trajectory 数据收集  
3. **完全依赖 Strands 的自然对话流程**

**根本原则**: rLLM + Strands 集成应该是**透明的**
- Strands: 处理对话、工具调用、记忆管理
- rLLM: 只记录轨迹数据用于训练

## Next Steps (Revised)
1. ❌ ~~Hard-coded loop detection~~ → ✅ **Better prompt engineering**
2. Add context/thought display back
3. Teach model to track its own progress  
4. Use conversation memory instead of action counting

## Test Command (v3 - Clean Architecture)
```bash
fish -c "conda activate rllm && cd examples/strands && python strands_rllm_browser_agent_v3.py"
```

## Architecture Comparison

### v2 (Wrong - Complex):
- ❌ 522 lines of manual history management
- ❌ Complex loop with `for step in range(max_steps)`  
- ❌ Manual `chat_messages` construction
- ❌ Hardcoded loop detection
- ❌ Token management logic
- ❌ Multiple dispatch paths

### v3 (Correct - Simple):
- ✅ 145 lines total
- ✅ Single call: `await agent.invoke_async(question)`
- ✅ Pure Strands conversation management
- ✅ Only RL tracking, no memory management
- ✅ Framework does what it's designed for

**Key Insight**: The complexity was **entirely unnecessary**. Strands already handles everything we were trying to rebuild.