/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import type { LlmConnection } from '@maka/core/llm-connections';
import type { LanguageModelV4ProviderTool, LanguageModelV4StreamPart } from '@ai-sdk/provider';
import { getAIModel } from '../model-factory.js';
import { lowerModelTools } from '../model-adapter.js';
import {
  createDeepSeekOpenResponsesExtensions,
  DEEPSEEK_OPEN_RESPONSES_WEB_SEARCH_EXTENSION_ID,
  openResponsesSupportsBareExtensionTypes,
  rewriteDeepSeekOpenResponsesIncomingValue,
  rewriteDeepSeekOpenResponsesOutgoingBody,
  usesDeepSeekOpenResponsesExtensions,
  wrapFetchForDeepSeekOpenResponsesExtensions,
} from '../deepseek-open-responses-extensions.js';
import { routeWebSearchTools } from '../native-web-search-tool.js';

function conn(providerType: LlmConnection['providerType'], slug = 'test'): LlmConnection {
  return {
    slug,
    name: slug,
    providerType,
    defaultModel: 'm',
    enabled: true,
    createdAt: 0,
    updatedAt: 0,
  };
}

function webSearchTool(): LanguageModelV4ProviderTool {
  const tools = lowerModelTools({
    WebSearch: { kind: 'provider', providerTool: { kind: 'openai-web-search' } },
  });
  return {
    ...(tools.WebSearch as object),
    type: 'provider',
    id: DEEPSEEK_OPEN_RESPONSES_WEB_SEARCH_EXTENSION_ID,
    name: 'WebSearch',
    args: { searchContextSize: 'medium' },
  };
}

function completedResponse(output: unknown[]): Record<string, unknown> {
  return {
    id: 'resp_deepseek_search',
    object: 'response',
    created_at: 1_700_000_000,
    model: 'deepseek-v4-flash',
    status: 'completed',
    output,
    usage: { input_tokens: 8, output_tokens: 4 },
  };
}

function sse(events: Array<Record<string, unknown>>): string {
  return events
    .map((event, index) => `data: ${JSON.stringify({ sequence_number: index, ...event })}\n\n`)
    .join('');
}

describe('DeepSeek Open Responses extension codecs', () => {
  test('registers against the compiled Open Responses search tool id', () => {
    const extensions = createDeepSeekOpenResponsesExtensions();
    assert.equal(extensions.length, 1);
    assert.equal(extensions[0]?.id, DEEPSEEK_OPEN_RESPONSES_WEB_SEARCH_EXTENSION_ID);
    assert.equal(usesDeepSeekOpenResponsesExtensions('deepseek'), true);
    assert.equal(usesDeepSeekOpenResponsesExtensions('alibaba-token-plan-cn'), false);
  });

  test('rewrites only allowlisted DeepSeek discriminators', () => {
    if (openResponsesSupportsBareExtensionTypes()) return;
    assert.deepEqual(
      rewriteDeepSeekOpenResponsesOutgoingBody({
        tools: [
          { type: 'openai:web_search' },
          { type: 'function', name: 'Read' },
          { type: 'openai.file_search' },
        ],
        tool_choice: { type: 'openai:web_search' },
        input: [
          { type: 'message', role: 'user', content: 'hi' },
          { type: 'openai:web_search_call', id: 'ws_1', status: 'completed' },
        ],
      }),
      {
        tools: [
          { type: 'web_search' },
          { type: 'function', name: 'Read' },
          { type: 'openai.file_search' },
        ],
        tool_choice: { type: 'web_search' },
        input: [
          { type: 'message', role: 'user', content: 'hi' },
          { type: 'web_search_call', id: 'ws_1', status: 'completed' },
        ],
      },
    );
    assert.deepEqual(
      rewriteDeepSeekOpenResponsesIncomingValue({
        type: 'response.web_search_call.in_progress',
        item_id: 'ws_1',
        item: { type: 'web_search_call', id: 'ws_1', status: 'in_progress' },
        output: [{ type: 'web_search_call', id: 'ws_1', status: 'completed' }],
      }),
      {
        type: 'openai:web_search_call.in_progress',
        item_id: 'ws_1',
        item: { type: 'openai:web_search_call', id: 'ws_1', status: 'in_progress' },
        output: [{ type: 'openai:web_search_call', id: 'ws_1', status: 'completed' }],
      },
    );
    assert.equal(
      (
        rewriteDeepSeekOpenResponsesIncomingValue({ type: 'web_search_2025_08_26' }) as {
          type: string;
        }
      ).type,
      'openai:web_search',
    );
  });

  test('wrapFetch rewrites only allowlisted discriminators on the wire', async () => {
    if (openResponsesSupportsBareExtensionTypes()) return;
    let sent: Record<string, unknown> | undefined;
    const fetch = wrapFetchForDeepSeekOpenResponsesExtensions(async (_url, init) => {
      sent = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return Response.json({
        output: [
          { type: 'web_search_call', id: 'ws_1', status: 'completed' },
          {
            type: 'function_call',
            id: 'fc_1',
            call_id: 'call_read',
            name: 'Read',
            arguments: '{}',
          },
        ],
      });
    });
    const response = await fetch('https://example.test/v1/responses', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        tools: [{ type: 'openai:web_search' }, { type: 'function', name: 'Read' }],
        input: [{ type: 'openai:web_search_call', id: 'ws_1', status: 'completed' }],
      }),
    });
    assert.deepEqual(sent?.tools, [{ type: 'web_search' }, { type: 'function', name: 'Read' }]);
    assert.deepEqual(
      ((await response.json()) as { output: Array<{ type: string }> }).output.map(
        (item) => item.type,
      ),
      ['openai:web_search_call', 'function_call'],
    );
  });

  test('encodes DeepSeek hosted search as a bare web_search tool', async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return Response.json(completedResponse([]));
    }) as unknown as typeof globalThis.fetch;
    const model = getAIModel({
      connection: conn('deepseek'),
      apiKey: 'test-key',
      modelId: 'deepseek-v4-flash',
      fetch,
    });
    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'search the web' }] }],
      tools: [webSearchTool()],
    });

    assert.deepEqual(bodies[0]?.tools, [{ type: 'web_search' }]);
    assert.equal(
      result.warnings?.some(
        (warning) =>
          warning.type === 'unsupported' &&
          warning.feature ===
            `provider-defined tool ${DEEPSEEK_OPEN_RESPONSES_WEB_SEARCH_EXTENSION_ID}`,
      ),
      false,
      JSON.stringify(result.warnings),
    );
  });

  test('omits unregistered provider tools with an explicit warning', async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return Response.json(completedResponse([]));
    }) as unknown as typeof globalThis.fetch;
    const model = getAIModel({
      connection: conn('deepseek'),
      apiKey: 'test-key',
      modelId: 'deepseek-v4-flash',
      fetch,
    });
    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'search files' }] }],
      tools: [
        webSearchTool(),
        { type: 'provider', id: 'openai.file_search', name: 'file_search', args: {} },
      ],
    });

    assert.deepEqual(bodies[0]?.tools, [{ type: 'web_search' }]);
    assert.equal(
      result.warnings?.some(
        (warning) =>
          warning.type === 'unsupported' &&
          warning.feature === 'provider-defined tool openai.file_search',
      ),
      true,
      JSON.stringify(result.warnings),
    );
  });

  test('decodes a completed hosted search item without entering the client tool loop', async () => {
    const fetch = (async () =>
      Response.json(
        completedResponse([
          {
            id: 'ws_opaque',
            type: 'web_search_call',
            status: 'completed',
            provider_trace: 'opaque-replay',
            action: { type: 'search', query: 'latest Maka', queries: ['latest Maka'] },
          },
          {
            id: 'msg_1',
            type: 'message',
            status: 'completed',
            role: 'assistant',
            content: [{ type: 'output_text', text: 'Maka shipped the feature.' }],
          },
        ]),
      )) as unknown as typeof globalThis.fetch;
    const model = getAIModel({
      connection: conn('deepseek'),
      apiKey: 'test-key',
      modelId: 'deepseek-v4-flash',
      fetch,
    });
    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'search' }] }],
      tools: [webSearchTool()],
    });

    const types = result.content.map((part) => part.type);
    assert.deepEqual(
      types.filter((type) => type === 'tool-call' || type === 'tool-result' || type === 'text'),
      ['tool-call', 'tool-result', 'text'],
    );
    const call = result.content.find((part) => part.type === 'tool-call');
    const searchResult = result.content.find((part) => part.type === 'tool-result');
    assert.equal(call && 'providerExecuted' in call ? call.providerExecuted : undefined, true);
    assert.equal(call && 'toolName' in call ? call.toolName : undefined, 'WebSearch');
    assert.match(JSON.stringify(call), /latest Maka/);
    assert.match(JSON.stringify(searchResult), /latest Maka/);
    assert.equal(
      result.finishReason.unified === 'stop' || result.finishReason.unified === 'tool-calls',
      true,
      JSON.stringify(result.finishReason),
    );
  });

  test('keeps mixed client and provider-executed tools in chronology', async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return Response.json(
        completedResponse([
          {
            id: 'ws_mixed',
            type: 'web_search_call',
            status: 'completed',
            action: { type: 'search', query: 'maka codecs' },
          },
          {
            id: 'fc_read',
            type: 'function_call',
            status: 'completed',
            call_id: 'call_read',
            name: 'Read',
            arguments: '{"path":"README.md"}',
          },
        ]),
      );
    }) as unknown as typeof globalThis.fetch;
    const model = getAIModel({
      connection: conn('deepseek'),
      apiKey: 'test-key',
      modelId: 'deepseek-v4-flash',
      fetch,
    });
    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'search then read' }] }],
      tools: [
        webSearchTool(),
        {
          type: 'function',
          name: 'Read',
          inputSchema: {
            type: 'object',
            properties: { path: { type: 'string' } },
            required: ['path'],
            additionalProperties: false,
          },
        },
      ],
    });

    assert.deepEqual(
      (bodies[0]?.tools as Array<Record<string, unknown>> | undefined)?.map((tool) => tool.type),
      ['web_search', 'function'],
    );
    const owned = result.content
      .filter((part) => part.type === 'tool-call' || part.type === 'tool-result')
      .map((part) => ({
        type: part.type,
        toolName: 'toolName' in part ? part.toolName : undefined,
        providerExecuted: 'providerExecuted' in part ? part.providerExecuted : undefined,
      }));
    assert.deepEqual(owned, [
      { type: 'tool-call', toolName: 'WebSearch', providerExecuted: true },
      { type: 'tool-result', toolName: 'WebSearch', providerExecuted: true },
      { type: 'tool-call', toolName: 'Read', providerExecuted: undefined },
    ]);
  });

  test('replays the original hosted search item exactly once', async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return Response.json(
        completedResponse([
          {
            id: 'ws_opaque',
            type: 'web_search_call',
            status: 'completed',
            provider_trace: 'opaque-replay',
            action: { type: 'open_page', url: 'https://maka.example/' },
          },
          {
            id: 'msg_1',
            type: 'message',
            status: 'completed',
            role: 'assistant',
            content: [{ type: 'output_text', text: 'Opened the page.' }],
          },
        ]),
      );
    }) as unknown as typeof globalThis.fetch;
    const model = getAIModel({
      connection: conn('deepseek'),
      apiKey: 'test-key',
      modelId: 'deepseek-v4-flash',
      fetch,
    });
    const first = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'open the docs' }] }],
      tools: [webSearchTool()],
    });
    await model.doGenerate({
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'open the docs' }] },
        { role: 'assistant', content: first.content as never },
        { role: 'user', content: [{ type: 'text', text: 'continue' }] },
      ],
      tools: [webSearchTool()],
    });

    const replayed = (bodies[1]?.input as Array<Record<string, unknown>> | undefined)?.filter(
      (item) => item.type === 'web_search_call',
    );
    assert.equal(replayed?.length, 1, JSON.stringify(bodies[1]?.input));
    assert.equal(replayed?.[0]?.id, 'ws_opaque');
    assert.equal(replayed?.[0]?.provider_trace, 'opaque-replay');
    assert.deepEqual(replayed?.[0]?.action, { type: 'open_page', url: 'https://maka.example/' });
  });

  test('streams hosted-search progress then finishes without a client tool call', async () => {
    const fetch = (async () =>
      new Response(
        sse([
          {
            type: 'response.created',
            response: { id: 'resp_stream', status: 'in_progress', output: [] },
          },
          {
            type: 'response.output_item.added',
            output_index: 0,
            item: { id: 'ws_stream', type: 'web_search_call', status: 'in_progress' },
          },
          { type: 'response.web_search_call.in_progress', item_id: 'ws_stream' },
          { type: 'response.web_search_call.searching', item_id: 'ws_stream' },
          {
            type: 'response.output_item.done',
            output_index: 0,
            item: {
              id: 'ws_stream',
              type: 'web_search_call',
              status: 'completed',
              action: { type: 'find_in_page', url: 'https://maka.example/', pattern: 'codec' },
            },
          },
          {
            type: 'response.output_item.added',
            output_index: 1,
            item: {
              id: 'msg_stream',
              type: 'message',
              status: 'in_progress',
              role: 'assistant',
              content: [],
            },
          },
          {
            type: 'response.content_part.added',
            item_id: 'msg_stream',
            output_index: 1,
            content_index: 0,
            part: { type: 'output_text', text: '' },
          },
          {
            type: 'response.output_text.delta',
            item_id: 'msg_stream',
            output_index: 1,
            content_index: 0,
            delta: 'Found the codec notes.',
          },
          {
            type: 'response.output_text.done',
            item_id: 'msg_stream',
            output_index: 1,
            content_index: 0,
            text: 'Found the codec notes.',
          },
          {
            type: 'response.content_part.done',
            item_id: 'msg_stream',
            output_index: 1,
            content_index: 0,
            part: { type: 'output_text', text: 'Found the codec notes.' },
          },
          {
            type: 'response.output_item.done',
            output_index: 1,
            item: {
              id: 'msg_stream',
              type: 'message',
              status: 'completed',
              role: 'assistant',
              content: [{ type: 'output_text', text: 'Found the codec notes.' }],
            },
          },
          {
            type: 'response.completed',
            response: {
              id: 'resp_stream',
              status: 'completed',
              output: [],
              usage: { input_tokens: 3, output_tokens: 2 },
            },
          },
        ]),
        { status: 200, headers: { 'content-type': 'text/event-stream' } },
      )) as unknown as typeof globalThis.fetch;
    const model = getAIModel({
      connection: conn('deepseek'),
      apiKey: 'test-key',
      modelId: 'deepseek-v4-flash',
      fetch,
    });
    const { stream } = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'find the codec' }] }],
      tools: [webSearchTool()],
    });
    const parts: LanguageModelV4StreamPart[] = [];
    for await (const part of stream) parts.push(part);

    assert.equal(
      parts.some(
        (part) =>
          part.type === 'tool-input-start' &&
          part.toolName === 'WebSearch' &&
          part.providerExecuted === true,
      ),
      true,
      JSON.stringify(parts.map((part) => part.type)),
    );
    assert.equal(
      parts.some(
        (part) =>
          part.type === 'tool-call' &&
          part.toolName === 'WebSearch' &&
          part.providerExecuted === true,
      ),
      true,
      JSON.stringify(
        parts.filter((part) => part.type === 'tool-call' || part.type === 'tool-result'),
      ),
    );
    assert.equal(
      parts.some((part) => part.type === 'tool-result' && part.toolName === 'WebSearch'),
      true,
    );
    assert.match(JSON.stringify(parts), /Found the codec notes/);
    const finish = parts.find((part) => part.type === 'finish');
    assert.ok(finish);
  });

  test('leaves generic Open Responses providers fail-closed for hosted search', async () => {
    const bodies: Record<string, unknown>[] = [];
    const fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return Response.json(completedResponse([]));
    }) as unknown as typeof globalThis.fetch;
    const model = getAIModel({
      connection: { ...conn('alibaba-token-plan-cn'), defaultModel: 'qwen3.8-max' },
      apiKey: 'test-key',
      modelId: 'qwen3.8-max',
      fetch,
    });
    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'search' }] }],
      tools: [webSearchTool()],
    });
    assert.equal(bodies[0]?.tools, undefined);
    assert.equal(
      result.warnings?.some(
        (warning) =>
          warning.type === 'unsupported' &&
          warning.feature ===
            `provider-defined tool ${DEEPSEEK_OPEN_RESPONSES_WEB_SEARCH_EXTENSION_ID}`,
      ),
      true,
      JSON.stringify(result.warnings),
    );
  });

  test('keeps Tavily and Anthropic-compatible DeepSeek routing off the Responses codec', () => {
    const tavily = {
      name: 'WebSearch',
      description: 'Tavily',
      parameters: {},
      impl: async () => undefined,
    };
    const routedTavily = routeWebSearchTools({
      tools: [tavily],
      settings: { enabled: true, defaultProvider: 'tavily' },
      connection: {
        slug: 'deepseek',
        providerType: 'deepseek',
        defaultModel: 'deepseek-v4-flash',
      },
      model: 'deepseek-v4-flash',
      tavilyReady: true,
    });
    assert.equal(routedTavily[0], tavily);

    const routedAnthropic = routeWebSearchTools({
      tools: [tavily],
      settings: { enabled: true, defaultProvider: 'model' },
      connection: {
        slug: 'anthropic-compatible',
        providerType: 'anthropic-compatible',
        defaultModel: 'deepseek-v4-flash',
        models: [{ id: 'deepseek-v4-flash', apiProtocol: 'anthropic-messages' }],
      },
      model: 'deepseek-v4-flash',
      tavilyReady: false,
    });
    assert.equal(routedAnthropic[0]?.providerTool?.kind, 'anthropic-web-search-20250305');
  });
});
