# 基于 TypeScript + Mastra AI 框架实现类 DeerFlow 项目的最佳实践

本文档介绍如何使用 TypeScript 技术栈的 [Mastra AI](https://mastra.ai) 框架实现一个类似 DeerFlow 的超级智能体系统，并提供架构映射、代码示例和最佳实践指南。

---

## 目录

1. [框架对比：LangGraph vs Mastra](#1-框架对比langgraph-vs-mastra)
2. [项目初始化](#2-项目初始化)
3. [架构映射](#3-架构映射)
4. [核心实现：Agent（智能体）](#4-核心实现agent智能体)
5. [工具系统（Tools）](#5-工具系统tools)
6. [工作流系统（Workflows）](#6-工作流系统workflows)
7. [记忆系统（Memory）](#7-记忆系统memory)
8. [MCP 集成](#8-mcp-集成)
9. [沙箱执行环境](#9-沙箱执行环境)
10. [网关 API 层](#10-网关-api-层)
11. [技能系统（Skills）](#11-技能系统skills)
12. [前端集成](#12-前端集成)
13. [项目目录结构推荐](#13-项目目录结构推荐)
14. [最佳实践总结](#14-最佳实践总结)

---

## 1. 框架对比：LangGraph vs Mastra

| 特性 | DeerFlow (Python + LangGraph) | Mastra 方案 (TypeScript) |
|------|-------------------------------|--------------------------|
| **语言** | Python 3.12+ | TypeScript / Node.js |
| **核心框架** | LangGraph 1.0+ | Mastra AI |
| **Agent 定义** | `CompiledGraph` | `Agent` class |
| **工具定义** | `@tool` decorator (LangChain) | `createTool()` |
| **工作流** | LangGraph StateGraph | Mastra `Workflow` |
| **记忆系统** | 自定义 JSON 文件 + LLM 提取 | Mastra `Memory` (LibSQL/PostgreSQL) |
| **MCP 集成** | `langchain-mcp-adapters` | Mastra MCP Client |
| **流式输出** | LangGraph SSE | Mastra `streamText` / SSE |
| **向量存储** | 可选 | 内置 `MastraVector` |
| **API 层** | FastAPI | Hono / Next.js API Routes |
| **运行时** | LangGraph Server | Mastra Server (`mastra dev`) |

### Mastra 核心优势

- **原生 TypeScript**：全栈类型安全，与 Next.js 前端无缝集成
- **内置 Memory**：开箱即用的对话记忆与语义搜索
- **Workflow 引擎**：基于步骤的确定性工作流，支持条件分支、并行执行
- **MCP 原生支持**：内置 MCP 客户端与服务器支持
- **可观测性**：内置追踪（OTel）和日志

---

## 2. 项目初始化

### 2.1 创建 Mastra 项目

```bash
# 使用官方脚手架工具
npx create-mastra@latest my-deerflow-ts

# 或手动初始化
mkdir my-deerflow-ts && cd my-deerflow-ts
pnpm init
pnpm add @mastra/core @mastra/memory @mastra/mcp
pnpm add hono @hono/node-server
pnpm add -D typescript @types/node tsx
```

### 2.2 基础配置

```typescript
// src/mastra/index.ts
import { Mastra } from "@mastra/core";
import { LibSQLStore } from "@mastra/libsql";
import { leadAgent } from "./agents/lead-agent";
import { researchWorkflow } from "./workflows/research";

export const mastra = new Mastra({
  agents: { leadAgent },
  workflows: { researchWorkflow },
  storage: new LibSQLStore({ url: "file:./data/mastra.db" }),
  // 可选：配置 OTel 可观测性
  telemetry: {
    serviceName: "my-deerflow-ts",
    enabled: process.env.NODE_ENV === "production",
  },
});
```

### 2.3 环境变量配置

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...

# 数据存储
DATABASE_URL=file:./data/mastra.db

# 可选工具
TAVILY_API_KEY=...
GITHUB_TOKEN=...
```

---

## 3. 架构映射

DeerFlow 的核心架构组件与 Mastra 的对应关系：

```
DeerFlow (Python)                    Mastra (TypeScript)
─────────────────────────────────    ─────────────────────────────────
make_lead_agent()           →        new Agent({ ... })
ThreadState                 →        Thread + Memory
Middleware Chain (11 个)    →        Agent hooks + Workflow steps
Sandbox Tools               →        createTool() + 自定义执行器
Subagent System             →        Agent-to-Agent calls / Workflows
Memory Middleware           →        Mastra Memory (自动)
MCP Integration             →        MastraMCPClient
Skills (SKILL.md)           →        Agent instructions + Workflows
Gateway API (FastAPI)       →        Hono / Next.js API Routes
LangGraph Server            →        mastra dev / 自定义 HTTP server
```

---

## 4. 核心实现：Agent（智能体）

### 4.1 主 Agent（对应 DeerFlow 的 lead_agent）

```typescript
// src/mastra/agents/lead-agent.ts
import { Agent } from "@mastra/core/agent";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { memory } from "../memory";
import {
  bashTool,
  readFileTool,
  writeFileTool,
  taskTool,
  clarificationTool,
  presentFileTool,
} from "../tools";

const model = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})("gpt-4o");

export const leadAgent = new Agent({
  name: "lead-agent",
  instructions: `
你是一个强大的 AI 超级助手，可以通过调用工具和子智能体来完成复杂任务。

## 核心能力
- 执行 Shell 命令和代码
- 读写文件和管理工作区
- 委托复杂任务给专门的子智能体
- 记忆用户偏好和历史上下文

## 工作原则
1. 优先理解用户意图，必要时通过 clarification 工具澄清需求
2. 将复杂任务分解为子任务，使用 task 工具委托执行
3. 生成文件后使用 present_file 工具展示结果
4. 保持简洁高效，避免不必要的中间确认

## 虚拟路径系统
- /mnt/user-data/workspace/ → 当前线程工作区
- /mnt/user-data/uploads/   → 用户上传的文件
- /mnt/user-data/outputs/   → 生成的输出文件
`,
  model,
  tools: {
    bash: bashTool,
    read_file: readFileTool,
    write_file: writeFileTool,
    task: taskTool,
    clarification: clarificationTool,
    present_file: presentFileTool,
  },
  memory,
});
```

### 4.2 子 Agent（对应 DeerFlow 的 subagents）

```typescript
// src/mastra/agents/sub-agents/general-purpose.ts
import { Agent } from "@mastra/core/agent";
import { createOpenAI } from "@ai-sdk/openai";
import { bashTool, readFileTool, writeFileTool } from "../../tools";

export const generalPurposeAgent = new Agent({
  name: "general-purpose",
  instructions: `
你是一个通用子智能体，负责执行具体的技术任务。
接收来自主智能体的明确指令并返回结构化结果。
`,
  model: createOpenAI({ apiKey: process.env.OPENAI_API_KEY })("gpt-4o"),
  tools: {
    bash: bashTool,
    read_file: readFileTool,
    write_file: writeFileTool,
  },
});
```

### 4.3 流式响应

```typescript
// src/api/threads/[threadId]/stream.ts (Next.js API Route)
import { mastra } from "@/mastra";

export async function POST(
  req: Request,
  { params }: { params: { threadId: string } }
) {
  const { message } = await req.json();
  const agent = mastra.getAgent("leadAgent");

  const stream = await agent.stream(message, {
    threadId: params.threadId,
    resourceId: "user-session",
    onFinish: async (result) => {
      // 异步后处理：标题生成、记忆提取等
      await processAfterCompletion(params.threadId, result);
    },
  });

  return stream.toDataStreamResponse();
}
```

---

## 5. 工具系统（Tools）

### 5.1 Bash 执行工具（对应 DeerFlow 的 sandbox bash tool）

```typescript
// src/mastra/tools/bash.ts
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { execSandbox } from "../sandbox";

export const bashTool = createTool({
  id: "bash",
  description: "在沙箱环境中执行 Shell 命令",
  inputSchema: z.object({
    command: z.string().describe("要执行的 Shell 命令"),
    workingDir: z.string().optional().describe("工作目录（虚拟路径）"),
    timeout: z.number().optional().default(30).describe("超时时间（秒）"),
  }),
  outputSchema: z.object({
    exitCode: z.number(),
    stdout: z.string(),
    stderr: z.string(),
  }),
  execute: async ({ context }) => {
    const { command, workingDir, timeout: timeoutSeconds } = context;
    const resolvedDir = resolveVirtualPath(workingDir);

    return await execSandbox(command, {
      cwd: resolvedDir,
      timeout: timeoutSeconds * 1000,
    });
  },
});
```

### 5.2 文件操作工具

```typescript
// src/mastra/tools/file-ops.ts
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import * as fs from "fs/promises";
import * as path from "path";
import { resolveVirtualPath } from "../sandbox/paths";

export const readFileTool = createTool({
  id: "read_file",
  description: "读取文件内容",
  inputSchema: z.object({
    path: z.string().describe("文件路径（支持虚拟路径 /mnt/user-data/...）"),
  }),
  outputSchema: z.object({
    content: z.string(),
    mimeType: z.string(),
  }),
  execute: async ({ context }) => {
    const realPath = resolveVirtualPath(context.path);
    const content = await fs.readFile(realPath, "utf-8");
    return { content, mimeType: detectMimeType(realPath) };
  },
});

export const writeFileTool = createTool({
  id: "write_file",
  description: "写入文件内容",
  inputSchema: z.object({
    path: z.string().describe("目标路径（支持虚拟路径）"),
    content: z.string().describe("文件内容"),
  }),
  outputSchema: z.object({ success: z.boolean(), path: z.string() }),
  execute: async ({ context }) => {
    const realPath = resolveVirtualPath(context.path);
    await fs.mkdir(path.dirname(realPath), { recursive: true });
    await fs.writeFile(realPath, context.content, "utf-8");
    return { success: true, path: context.path };
  },
});
```

### 5.3 任务委托工具（对应 DeerFlow 的 task_tool）

```typescript
// src/mastra/tools/task.ts
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { mastra } from "../index";

const SUBAGENT_TIMEOUT_MS = 15 * 60 * 1000; // 15 分钟

export const taskTool = createTool({
  id: "task",
  description: "将复杂任务委托给专门的子智能体执行",
  inputSchema: z.object({
    description: z.string().describe("任务简要描述"),
    instructions: z.string().describe("详细执行指令"),
    agentType: z
      .enum(["general-purpose", "bash", "research"])
      .optional()
      .default("general-purpose")
      .describe("子智能体类型"),
  }),
  outputSchema: z.object({
    taskId: z.string(),
    status: z.enum(["completed", "failed"]),
    result: z.string(),
  }),
  execute: async ({ context, threadId }) => {
    const taskId = crypto.randomUUID();
    const subAgent = mastra.getAgent(context.agentType);

    try {
      const result = await subAgent.generate(context.instructions, {
        threadId: `${threadId}-subtask-${taskId}`,
        abortSignal: AbortSignal.timeout(SUBAGENT_TIMEOUT_MS),
      });

      return {
        taskId,
        status: "completed" as const,
        result: result.text,
      };
    } catch (error) {
      return {
        taskId,
        status: "failed" as const,
        result: String(error),
      };
    }
  },
});
```

### 5.4 澄清工具（对应 DeerFlow 的 clarification_tool）

```typescript
// src/mastra/tools/clarification.ts
import { createTool } from "@mastra/core/tools";
import { z } from "zod";

export const clarificationTool = createTool({
  id: "clarification",
  description: "当任务不明确时，向用户请求澄清",
  inputSchema: z.object({
    question: z.string().describe("需要澄清的问题"),
    options: z.array(z.string()).optional().describe("可选的预设选项"),
  }),
  outputSchema: z.object({
    clarificationRequested: z.boolean(),
  }),
  execute: async ({ context }) => {
    // 在流式响应中，这会触发前端展示澄清 UI
    // 通过 SSE 事件通知前端需要用户输入
    return { clarificationRequested: true };
  },
});
```

---

## 6. 工作流系统（Workflows）

Mastra Workflow 是实现 DeerFlow 复杂多步骤任务的关键，对应 DeerFlow 的 subagent 并发执行和 middleware chain 逻辑。

### 6.1 深度研究工作流

```typescript
// src/mastra/workflows/deep-research.ts
import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

const planStep = createStep({
  id: "plan",
  description: "分析问题并制定研究计划",
  inputSchema: z.object({ topic: z.string() }),
  outputSchema: z.object({
    queries: z.array(z.string()),
    outline: z.string(),
  }),
  execute: async ({ context, mastra }) => {
    const agent = mastra!.getAgent("leadAgent");
    const result = await agent.generate(
      `为以下主题制定详细的研究计划，输出 JSON 格式：
       主题：${context.topic}
       输出格式：{ queries: string[], outline: string }`
    );
    return JSON.parse(result.text);
  },
});

const searchStep = createStep({
  id: "search",
  description: "并行执行多个搜索查询",
  inputSchema: z.object({ queries: z.array(z.string()) }),
  outputSchema: z.object({ searchResults: z.array(z.string()) }),
  execute: async ({ context }) => {
    const results = await Promise.all(
      context.queries.map((q) => searchWeb(q))
    );
    return { searchResults: results.flat() };
  },
});

const synthesizeStep = createStep({
  id: "synthesize",
  description: "综合搜索结果生成最终报告",
  inputSchema: z.object({
    outline: z.string(),
    searchResults: z.array(z.string()),
  }),
  outputSchema: z.object({ report: z.string() }),
  execute: async ({ context, mastra }) => {
    const agent = mastra!.getAgent("leadAgent");
    const result = await agent.generate(
      `基于以下大纲和搜索结果生成详细报告：
       大纲：${context.outline}
       搜索结果：${context.searchResults.join("\n\n")}`
    );
    return { report: result.text };
  },
});

export const deepResearchWorkflow = createWorkflow({
  id: "deep-research",
  inputSchema: z.object({ topic: z.string() }),
  outputSchema: z.object({ report: z.string() }),
})
  .then(planStep)
  .then(searchStep)
  .then(synthesizeStep)
  .commit();
```

### 6.2 并行子任务工作流（对应 DeerFlow 的并行 subagent 执行）

```typescript
// src/mastra/workflows/parallel-tasks.ts
import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

const SUBAGENT_TIMEOUT_MS = 15 * 60 * 1000; // 15 分钟
const TaskSchema = z.object({ id: z.string(), instruction: z.string() });

const executeTasksStep = createStep({
  id: "execute-parallel-tasks",
  description: "并行执行多个子任务",
  inputSchema: z.object({ tasks: z.array(TaskSchema) }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        id: z.string(),
        status: z.enum(["completed", "failed"]),
        result: z.string(),
      })
    ),
  }),
  execute: async ({ context, mastra }) => {
    // 最多同时运行 3 个子任务（对应 DeerFlow 的 MAX_SUBAGENT_CONCURRENCY=3）
    const MAX_CONCURRENT = 3;
    const results = [];

    for (let i = 0; i < context.tasks.length; i += MAX_CONCURRENT) {
      const batch = context.tasks.slice(i, i + MAX_CONCURRENT);
      const batchResults = await Promise.allSettled(
        batch.map(async (task) => {
          const agent = mastra!.getAgent("generalPurposeAgent");
          const result = await agent.generate(task.instruction, {
            abortSignal: AbortSignal.timeout(SUBAGENT_TIMEOUT_MS),
          });
          return { id: task.id, status: "completed" as const, result: result.text };
        })
      );

      results.push(
        ...batchResults.map((r, idx) =>
          r.status === "fulfilled"
            ? r.value
            : { id: batch[idx].id, status: "failed" as const, result: String(r.reason) }
        )
      );
    }

    return { results };
  },
});
```

---

## 7. 记忆系统（Memory）

### 7.1 配置 Mastra Memory

```typescript
// src/mastra/memory/index.ts
import { Memory } from "@mastra/memory";
import { LibSQLStore } from "@mastra/libsql";
import { LibSQLVector } from "@mastra/libsql";
import { createOpenAI } from "@ai-sdk/openai";

const embeddingModel = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
}).embedding("text-embedding-3-small");

export const memory = new Memory({
  storage: new LibSQLStore({
    url: process.env.DATABASE_URL ?? "file:./data/memory.db",
  }),
  vector: new LibSQLVector({
    connectionUrl: process.env.DATABASE_URL ?? "file:./data/memory.db",
  }),
  embedder: embeddingModel,
  options: {
    // 最近 N 条消息放入上下文
    lastMessages: 20,
    // 语义搜索召回 Top-K 相关记忆
    semanticRecall: {
      topK: 5,
      messageRange: { before: 2, after: 2 },
    },
    // 工作记忆：在系统提示中注入的持久化用户上下文
    workingMemory: {
      enabled: true,
      template: `
# 用户工作记忆

## 用户概要
<!-- 姓名、职业、工作领域 -->

## 近期关注
<!-- 最近讨论的主要主题 -->

## 用户偏好
<!-- 沟通风格、技术偏好等 -->

## 重要事实
<!-- 关于用户的重要已知事实 -->
`,
    },
  },
});
```

### 7.2 在 API 路由中使用记忆

```typescript
// src/api/chat/route.ts
import { mastra } from "@/mastra";

export async function POST(req: Request) {
  const { message, threadId, userId } = await req.json();
  const agent = mastra.getAgent("leadAgent");

  // Mastra 自动处理记忆的读取和写入
  const stream = await agent.stream(message, {
    threadId,
    resourceId: userId, // 用于跨线程记忆共享
  });

  return stream.toDataStreamResponse();
}
```

---

## 8. MCP 集成

### 8.1 MCP 客户端配置（对应 DeerFlow 的 extensions_config.json）

```typescript
// src/mastra/mcp/index.ts
import { MCPClient } from "@mastra/mcp";

export const mcpClient = new MCPClient({
  servers: {
    github: {
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-github"],
      env: {
        GITHUB_TOKEN: process.env.GITHUB_TOKEN ?? "",
      },
    },
    filesystem: {
      command: "npx",
      args: [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/mnt/user-data/workspace",
      ],
    },
    // HTTP 类型 MCP 服务器
    custom: {
      url: "http://localhost:8080/mcp",
      type: "http",
    },
  },
});
```

### 8.2 将 MCP 工具注入到 Agent

```typescript
// src/mastra/agents/lead-agent.ts（更新版）
import { Agent } from "@mastra/core/agent";
import { mcpClient } from "../mcp";

export async function createLeadAgent() {
  // 动态获取所有 MCP 工具
  const mcpTools = await mcpClient.getTools();

  return new Agent({
    name: "lead-agent",
    instructions: "...",
    model,
    tools: {
      ...builtinTools,
      ...mcpTools, // MCP 工具自动集成
    },
    memory,
  });
}
```

### 8.3 动态 MCP 配置更新（对应 DeerFlow 的 /api/mcp/config）

```typescript
// src/api/mcp/config/route.ts
import { mastra } from "@/mastra";
import { mcpClient } from "@/mastra/mcp";

export async function GET() {
  const config = await mcpClient.getConfig();
  return Response.json(config);
}

export async function PUT(req: Request) {
  const newConfig = await req.json();
  await mcpClient.updateConfig(newConfig);
  // 重新创建 Agent 以应用新的 MCP 工具
  await mastra.reloadAgent("leadAgent");
  return Response.json({ success: true });
}
```

---

## 9. 沙箱执行环境

### 9.1 本地沙箱（对应 DeerFlow 的 LocalSandboxProvider）

```typescript
// src/mastra/sandbox/local.ts
import { exec } from "child_process";
import { promisify } from "util";
import * as path from "path";
import * as fs from "fs/promises";

const execAsync = promisify(exec);

// 虚拟路径到真实路径的映射
const VIRTUAL_PATH_MAP: Record<string, string> = {
  "/mnt/user-data/workspace": "./data/workspace",
  "/mnt/user-data/uploads": "./data/uploads",
  "/mnt/user-data/outputs": "./data/outputs",
  "/mnt/skills": "./skills",
};

export function resolveVirtualPath(virtualPath: string): string {
  for (const [virtual, real] of Object.entries(VIRTUAL_PATH_MAP)) {
    if (virtualPath.startsWith(virtual)) {
      return path.join(real, virtualPath.slice(virtual.length));
    }
  }
  return virtualPath;
}

export async function execSandbox(
  command: string,
  options: { cwd?: string; timeout?: number } = {}
) {
  const { cwd = "./data/workspace", timeout = 30000 } = options;

  await fs.mkdir(cwd, { recursive: true });

  try {
    const { stdout, stderr } = await execAsync(command, {
      cwd,
      timeout,
      maxBuffer: 10 * 1024 * 1024, // 10MB
    });
    return { exitCode: 0, stdout, stderr };
  } catch (error: any) {
    return {
      exitCode: error.code ?? 1,
      stdout: error.stdout ?? "",
      stderr: error.stderr ?? String(error.message),
    };
  }
}
```

### 9.2 Docker 沙箱（对应 DeerFlow 的 AioSandboxProvider）

```typescript
// src/mastra/sandbox/docker.ts
import Docker from "dockerode";

const docker = new Docker({ socketPath: "/var/run/docker.sock" });

export class DockerSandbox {
  private container: Docker.Container | null = null;
  private threadId: string;

  constructor(threadId: string) {
    this.threadId = threadId;
  }

  async init() {
    this.container = await docker.createContainer({
      Image: "python:3.12-slim",
      name: `sandbox-${this.threadId}`,
      Cmd: ["sleep", "infinity"],
      HostConfig: {
        Memory: 512 * 1024 * 1024, // 512MB
        CpuPeriod: 100000,
        CpuQuota: 50000, // 50% CPU
        NetworkMode: "none", // 无网络访问
        Binds: [
          `./data/workspace/${this.threadId}:/mnt/user-data/workspace:rw`,
          `./skills:/mnt/skills:ro`,
        ],
      },
    });
    await this.container.start();
  }

  async exec(command: string): Promise<{ exitCode: number; output: string }> {
    if (!this.container) throw new Error("Container not initialized");

    const exec = await this.container.exec({
      Cmd: ["bash", "-c", command],
      AttachStdout: true,
      AttachStderr: true,
    });

    const stream = await exec.start({ hijack: true, stdin: false });
    const output = await collectStream(stream);
    const inspect = await exec.inspect();

    return { exitCode: inspect.ExitCode ?? 0, output };
  }

  async cleanup() {
    if (this.container) {
      await this.container.stop({ t: 1 });
      await this.container.remove();
    }
  }
}
```

---

## 10. 网关 API 层

### 10.1 使用 Hono 构建 API（对应 DeerFlow 的 FastAPI Gateway）

```typescript
// src/server/index.ts
import { Hono } from "hono";
import { cors } from "hono/cors";
import { stream } from "hono/streaming";
import { mastra } from "@/mastra";

const app = new Hono();

app.use("*", cors());

// ===== 模型列表 =====
app.get("/api/models", async (c) => {
  return c.json({
    models: [
      { id: "gpt-4o", name: "GPT-4o", provider: "openai" },
      { id: "claude-3-5-sonnet", name: "Claude 3.5 Sonnet", provider: "anthropic" },
      { id: "deepseek-v3", name: "DeepSeek V3", provider: "deepseek" },
    ],
  });
});

// ===== 线程管理 =====
app.post("/api/threads/:threadId/chat", async (c) => {
  const { threadId } = c.req.param();
  const { message, resourceId } = await c.req.json();
  const agent = mastra.getAgent("leadAgent");

  return stream(c, async (s) => {
    const result = await agent.stream(message, {
      threadId,
      resourceId,
    });

    for await (const chunk of result.textStream) {
      await s.write(`data: ${JSON.stringify({ text: chunk })}\n\n`);
    }

    await s.write("data: [DONE]\n\n");
  });
});

// ===== 文件上传 =====
app.post("/api/threads/:threadId/uploads", async (c) => {
  const { threadId } = c.req.param();
  const formData = await c.req.formData();
  const file = formData.get("file") as File;

  const uploadPath = `./data/uploads/${threadId}/${file.name}`;
  await writeUploadedFile(uploadPath, file);

  // 自动转换 PDF/Word/Excel 为 Markdown（使用 markitdown 等库）
  const markdownPath = await convertToMarkdown(uploadPath);

  return c.json({ path: markdownPath, originalName: file.name });
});

// ===== 技能管理 =====
app.get("/api/skills", async (c) => {
  const skills = await loadAllSkills("./skills");
  return c.json({ skills });
});

// ===== 记忆管理 =====
app.get("/api/memory/:resourceId", async (c) => {
  const { resourceId } = c.req.param();
  if (!mastra.memory) {
    return c.json({ error: "Memory is not configured" }, 503);
  }
  // Mastra Memory 提供线程和资源维度的记忆查询
  const threads = await mastra.memory.getThreadsByResourceId({ resourceId });
  return c.json({ threads });
});

export default app;
```

### 10.2 与 Next.js 集成（全栈方案）

```typescript
// src/app/api/chat/route.ts (Next.js App Router)
import { mastra } from "@/mastra";
import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  const { message, threadId, userId } = await req.json();
  const agent = mastra.getAgent("leadAgent");

  const result = await agent.stream(message, {
    threadId,
    resourceId: userId,
  });

  // 兼容 Vercel AI SDK useChat hook
  return result.toDataStreamResponse({
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    },
  });
}
```

---

## 11. 技能系统（Skills）

### 11.1 技能定义（对应 DeerFlow 的 SKILL.md）

Mastra 中的"技能"可以通过以下几种方式实现：

**方式一：Agent Instructions（简单技能）**

```typescript
// src/mastra/skills/deep-research.ts
export const deepResearchSkill = `
## 深度研究技能

当用户请求深度研究时，按以下流程执行：

1. **问题分析**：理解研究主题，识别关键问题
2. **制定计划**：列出 3-5 个核心研究问题
3. **信息收集**：使用搜索工具获取最新资料
4. **内容分析**：从多个角度分析收集的信息
5. **报告生成**：输出结构化的 Markdown 研究报告

报告格式：
- 执行摘要
- 详细分析（各部分有数据支撑）
- 结论与建议
- 参考资料
`;
```

**方式二：Workflow（复杂技能）**

```typescript
// src/mastra/skills/chart-visualization.ts
import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

export const chartVisualizationWorkflow = createWorkflow({
  id: "chart-visualization",
  inputSchema: z.object({
    data: z.string().describe("要可视化的数据（CSV/JSON）"),
    chartType: z.enum(["bar", "line", "pie", "scatter"]).optional(),
  }),
  outputSchema: z.object({ chartPath: z.string() }),
})
  .then(parseDataStep)
  .then(selectChartTypeStep)
  .then(generateCodeStep)
  .then(executeAndSaveStep)
  .commit();
```

### 11.2 技能加载器

```typescript
// src/mastra/skills/loader.ts
import * as fs from "fs/promises";
import * as path from "path";

interface Skill {
  name: string;
  description: string;
  content: string;
  path: string;
}

export async function loadAllSkills(skillsDir: string): Promise<Skill[]> {
  const skills: Skill[] = [];

  async function scanDir(dir: string) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await scanDir(fullPath);
      } else if (entry.name === "SKILL.md") {
        const content = await fs.readFile(fullPath, "utf-8");
        const skill = parseSkillFile(content, fullPath);
        if (skill) skills.push(skill);
      }
    }
  }

  await scanDir(skillsDir);
  return skills;
}

function parseSkillFile(content: string, filePath: string): Skill | null {
  // 解析 YAML frontmatter
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return null;

  const frontmatter = parseYaml(match[1]);
  return {
    name: frontmatter.name,
    description: frontmatter.description,
    content: match[2],
    path: filePath,
  };
}
```

---

## 12. 前端集成

DeerFlow 的 Next.js 前端可以与 Mastra 后端直接集成，几乎无需修改。

### 12.1 替换 LangGraph SDK 为 Vercel AI SDK

```typescript
// src/hooks/use-chat-stream.ts
import { useChat } from "ai/react";

export function useChatStream(threadId: string, userId: string) {
  return useChat({
    api: "/api/chat",
    body: { threadId, userId },
    // 与 DeerFlow 前端兼容的消息格式
    onResponse: (response) => {
      if (!response.ok) {
        console.error("Chat error:", response.statusText);
      }
    },
    onFinish: (message) => {
      // 处理完成事件
    },
    onError: (error) => {
      console.error("Stream error:", error);
    },
  });
}
```

### 12.2 SSE 事件处理

```typescript
// src/core/stream-handler.ts
// Mastra 的流式输出格式与 Vercel AI SDK 兼容
// 可以直接使用 DeerFlow 前端的现有 SSE 处理逻辑

export function parseAgentEvent(data: string) {
  try {
    const event = JSON.parse(data);
    switch (event.type) {
      case "text-delta":
        return { type: "text", content: event.textDelta };
      case "tool-call":
        return { type: "tool_call", tool: event.toolName, args: event.args };
      case "tool-result":
        return { type: "tool_result", result: event.result };
      case "finish":
        return { type: "done" };
      default:
        return null;
    }
  } catch {
    return null;
  }
}
```

---

## 13. 项目目录结构推荐

```
my-deerflow-ts/
├── src/
│   ├── mastra/                     # Mastra 核心配置
│   │   ├── index.ts                # Mastra 实例
│   │   ├── agents/                 # Agent 定义
│   │   │   ├── lead-agent.ts       # 主 Agent
│   │   │   └── sub-agents/         # 子 Agent
│   │   ├── tools/                  # 工具定义
│   │   │   ├── bash.ts
│   │   │   ├── file-ops.ts
│   │   │   ├── task.ts
│   │   │   ├── clarification.ts
│   │   │   └── web-search.ts
│   │   ├── workflows/              # 工作流定义
│   │   │   ├── deep-research.ts
│   │   │   ├── parallel-tasks.ts
│   │   │   └── report-generation.ts
│   │   ├── memory/                 # 记忆系统配置
│   │   │   └── index.ts
│   │   ├── mcp/                    # MCP 集成
│   │   │   └── index.ts
│   │   └── sandbox/                # 沙箱执行
│   │       ├── local.ts
│   │       └── docker.ts
│   ├── app/                        # Next.js App Router
│   │   ├── api/
│   │   │   ├── chat/route.ts
│   │   │   ├── models/route.ts
│   │   │   ├── mcp/config/route.ts
│   │   │   ├── skills/route.ts
│   │   │   └── memory/route.ts
│   │   └── (workspace)/            # 前端页面
│   ├── components/                 # React 组件
│   └── hooks/                      # React Hooks
├── skills/                         # 技能文件（复用 DeerFlow 格式）
│   ├── public/
│   └── custom/
├── data/                           # 运行时数据（gitignore）
│   ├── workspace/
│   ├── uploads/
│   ├── outputs/
│   └── memory.db
├── docker/
│   └── docker-compose.yml
├── package.json
├── tsconfig.json
└── .env
```

---

## 14. 最佳实践总结

### 14.1 类型安全

```typescript
// 使用 Zod 强制定义所有 Tool 的输入输出 Schema
// 这确保了类型安全并提供了清晰的 API 文档
export const myTool = createTool({
  id: "my_tool",
  inputSchema: z.object({ /* 严格定义 */ }),
  outputSchema: z.object({ /* 严格定义 */ }),
  execute: async ({ context }) => {
    // TypeScript 自动推断 context 类型
  },
});
```

### 14.2 错误处理策略

```typescript
// 在工具和工作流中统一错误处理
export const safeBashTool = createTool({
  id: "bash",
  // ...
  execute: async ({ context }) => {
    try {
      return await execSandbox(context.command);
    } catch (error) {
      // 返回结构化错误而不是抛出异常
      // 让 Agent 能够理解错误并决定如何处理
      return {
        exitCode: 1,
        stdout: "",
        stderr: `执行失败: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  },
});
```

### 14.3 线程隔离

```typescript
// 每个对话线程应使用独立的工作目录
// 使用 threadId 作为隔离键
export function getThreadWorkspace(threadId: string) {
  return {
    workspace: `./data/workspace/${threadId}`,
    uploads: `./data/uploads/${threadId}`,
    outputs: `./data/outputs/${threadId}`,
  };
}
```

### 14.4 流量控制与超时

```typescript
// 为长时间运行的任务设置合理的超时
const SUBAGENT_TIMEOUT_MS = 15 * 60 * 1000; // 15 分钟
const MAX_CONCURRENT_TASKS = 3; // 最大并发子任务数

// 使用 AbortSignal 控制超时
const result = await agent.generate(instruction, {
  abortSignal: AbortSignal.timeout(SUBAGENT_TIMEOUT_MS),
});
```

### 14.5 可观测性

```typescript
// 使用 Mastra 内置的 OTel 追踪
export const mastra = new Mastra({
  // ...
  telemetry: {
    serviceName: "my-deerflow-ts",
    sampling: { type: "ratio", probability: 0.1 }, // 采样 10%
    export: {
      type: "otlp",
      endpoint: process.env.OTLP_ENDPOINT,
    },
  },
  logger: {
    name: "my-deerflow-ts",
    level: process.env.NODE_ENV === "production" ? "warn" : "debug",
  },
});
```

### 14.6 配置管理

```typescript
// 使用环境变量 + 配置文件的组合
// 对应 DeerFlow 的 config.yaml + extensions_config.json
import { z } from "zod";

const ConfigSchema = z.object({
  models: z.object({
    default: z.string().default("gpt-4o"),
    reasoning: z.string().default("o1-mini"),
  }),
  sandbox: z.object({
    type: z.enum(["local", "docker"]).default("local"),
    timeout: z.number().default(30),
  }),
  memory: z.object({
    enabled: z.boolean().default(true),
    maxLastMessages: z.number().default(20),
  }),
});

export const config = ConfigSchema.parse({
  models: {
    default: process.env.DEFAULT_MODEL,
    reasoning: process.env.REASONING_MODEL,
  },
  // ...
});
```

### 14.7 技能系统复用

DeerFlow 的 `skills/` 目录使用标准 Markdown 格式，**可以直接复用**无需修改。只需更新加载器：

```typescript
// 加载 DeerFlow 格式的 SKILL.md 技能文件
const skills = await loadAllSkills("./skills");
const skillInstructions = skills
  .filter((s) => s.enabled)
  .map((s) => `## 技能：${s.name}\n${s.description}\n\n${s.content}`)
  .join("\n\n---\n\n");

// 注入到 Agent instructions 中
const agent = new Agent({
  instructions: `${baseInstructions}\n\n# 可用技能\n\n${skillInstructions}`,
  // ...
});
```

---

## 参考资源

- [Mastra AI 官方文档](https://mastra.ai/docs)
- [Mastra GitHub](https://github.com/mastra-ai/mastra)
- [Vercel AI SDK 文档](https://sdk.vercel.ai/docs)
- [Model Context Protocol 规范](https://modelcontextprotocol.io)
- [DeerFlow 架构文档](../backend/docs/ARCHITECTURE.md)
- [DeerFlow 后端文档](../backend/docs/README.md)
