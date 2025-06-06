# Self-Aware Agent Architecture

The Agent System is an adaptive AI architecture that autonomously generates evaluation criteria, produces multiple response candidates, and iteratively improves through confidence-based self-correction. The system leverages LLM-as-a-judge mechanisms combined with log probability confidence scoring to deliver high-quality outputs while continuously enhancing its own evaluation capabilities and prompt strategies.

## Core Architecture Components

### 1. **Response Generation Pipeline**

- Agent creates 3-5 response candidates for each user prompt
- Each response includes explicit chain-of-thought reasoning
- Responses tagged with initial confidence estimates
- Parallel generation paths for diverse output exploration

#### 2. **LLM-as-Judge Meta-Evaluation System**

- Dynamically creates evaluation criteria based on task context
- Multiple specialized judge models for different evaluation aspects
- Extracts confidence scores from model token probabilities
- Configurable confidence thresholds (e.g., >90% auto-accept, 50-90% human review, <50% auto-reject)

#### 3. **Agent Corrective Logic Engine**

- Identifies failure modes and success patterns
- Creates new evaluation criteria based on historical performance
- Generates improved prompts for both response generation and evaluation
- Develops specialized evaluation tools for domain-specific tasks

#### 4. **Memory & Knowledge Management**

- **Episodic Memory**: Stores interaction history and performance metrics
- **Semantic Memory**: Maintains knowledge about user preferences and successful patterns
- **Vector Embeddings**: MongoDB Atlas vector search for semantic similarity matching

### Architecture Flow

```
User Input → Response Generator → Multiple Candidates → Judge Evaluation → 
Confidence Analysis → Threshold Decision → Output/Correction Loop → 
Memory Storage → Pattern Analysis → Criteria/Prompt Updates
```

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ff6b6b', 'primaryTextColor': '#000', 'primaryBorderColor': '#ff6b6b', 'lineColor': '#333', 'secondaryColor': '#4ecdc4', 'tertiaryColor': '#45b7d1'}}}%%

graph TD
  UserPrompt[User Prompt Input] --> ResponseGen[Response Generation Pipeline]
  
  ResponseGen --> Response1[Response 1<br/>• Confidence Score<br/>• Message Content]
  ResponseGen --> Response2[Response 2<br/>• Confidence Score<br/>• Message Content]
  ResponseGen --> Response3[Response 3<br/>• Confidence Score<br/>• Message Content]
  
  Response1 --> JudgeSystem[LLM-as-Judge<br/>Meta-Evaluation]
  Response2 --> JudgeSystem
  Response3 --> JudgeSystem
  
  JudgeSystem --> LogProbs[Log Probabilities<br/>Confidence Analysis]
  LogProbs --> ConfidenceAvg[Confidence Average<br/>Calculation]
  
  ConfidenceAvg --> Threshold{Confidence<br/>Threshold<br/>Analysis}
  
  Threshold -->|>90% Auto-Accept| Output[Final Output<br/>Delivery]
  Threshold -->|50-90% Review| HumanReview[Human Review<br/>Required]
  Threshold -->|<50% Reject| AgentCorrection[Agent Corrective<br/>Logic Engine]
  
  AgentCorrection --> CriteriaGen[Dynamic Criteria<br/>Generation]
  AgentCorrection --> PromptOpt[Prompt Strategy<br/>Optimization]
  AgentCorrection --> ToolCreation[Specialized Tool<br/>Development]
  
  CriteriaGen --> MemoryStore[Memory & Knowledge<br/>Management System]
  PromptOpt --> MemoryStore
  ToolCreation --> MemoryStore
  
  MemoryStore --> EpisodicMem[Episodic Memory<br/>Interaction History]
  MemoryStore --> SemanticMem[Semantic Memory<br/>User Preferences]
  MemoryStore --> VectorStore[Vector Embeddings<br/>MongoDB Atlas]
  
  HumanReview --> Feedback[User Feedback<br/>Upvote/Downvote]
  Output --> Feedback
  
  Feedback --> MemoryStore
  
  style UserPrompt fill:#e1f5fe
  style ResponseGen fill:#f3e5f5
  style JudgeSystem fill:#fff3e0
  style Threshold fill:#f1f8e9
  style AgentCorrection fill:#ffebee
  style MemoryStore fill:#f9fbe7
```

</br>

#### Technical Stack

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#2196f3', 'primaryTextColor': '#000', 'primaryBorderColor': '#2196f3'}}}%%

graph LR
  subgraph "Frontend Layer"
    UI[Real-time Confidence Display]
  end
  
  subgraph "API Layer"
    FastAPI[FastAPI Server<br/>• Async Processing]
  end
  
  subgraph "Orchestration Layer"
    LangGraph[LangGraph<br/>• State Machine Workflow<br/>• Agent Coordination<br/>]
    LangMem[LangMem<br/>• Long-term Memory<br/>• Cross-session Context<br/>• Knowledge Persistence]
  end
  
  subgraph "ML Layer"
    PrimaryLLM[GPT-4/Claude<br/>• Response Generation<br/>• Multi-candidate Output]
    JudgeLLM[Judge LLM<br/>• Quality Evaluation<br/>• Confidence Scoring<br/>• Meta-assessment]
    Embeddings[Text Embeddings<br/>• Semantic Similarity<br/>• Vector Representations]
  end
  
  subgraph "Storage Layer"
    MongoDB[MongoDB Atlas<br/>• Vector Search<br/>• Semantic Indexing]
  end
  
  UI --> FastAPI
  FastAPI --> LangGraph
  LangGraph --> LangMem
  LangGraph --> PrimaryLLM
  LangGraph --> JudgeLLM
  LangMem --> MongoDB
  Embeddings --> MongoDB
  
  style UI fill:#e3f2fd
  style FastAPI fill:#e8f5e8
  style LangGraph fill:#fff3e0
  style LangMem fill:#f3e5f5
  style PrimaryLLM fill:#ffebee
  style JudgeLLM fill:#f1f8e9
  style MongoDB fill:#e1f5fe
```

#### LangGraph Nodes

<img src="./agent/graph.png" width=450 alt="LangGraph StateGraph" />

- GPT-4/Claude for response generation
- Separate model instance for evaluation tasks
- Log probability extraction and normalization
- Text embedding models for semantic search

#### **Performance KPIs**

- Response quality scores (human evaluation baseline)
- Confidence calibration accuracy (predicted vs. actual quality)
- Self-improvement rate (performance gains over time)
- User satisfaction ratings

#### **Technical Metrics**

- Response latency (<2s for generation, <1s for evaluation)
- Confidence score accuracy (±10% of human assessment)
- Memory retrieval speed (<100ms for semantic search)
- System uptime and reliability (99.9% target)

#### **Quality Assurance**

- Human oversight for low-confidence decisions
- Regular evaluation of judge performance
- Fallback mechanisms for system failures
- Audit trails for all decisions and improvements
