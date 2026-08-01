package openai

import (
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
)

func TestStabilizeCachePrefix(t *testing.T) {
	req := &Request{Messages: []Message{
		{
			Role: "system",
			Content: MessageContent{Content: lo.ToPtr(
				"x-anthropic-billing-header: cc_version=2.1.177.c0b; cc_entrypoint=cli; cch=abc12;",
			)},
		},
		{
			Role: "system",
			Content: MessageContent{Content: lo.ToPtr(
				"x-anthropic-billing-header: cc_version=2.1.177.c0b; cch=abc12; You are Claude Code.",
			)},
		},
		{Role: "system", Content: MessageContent{Content: lo.ToPtr("Stable system prompt")}},
		{Role: "user", Content: MessageContent{Content: lo.ToPtr("First turn")}},
		{Role: "SYSTEM", Content: MessageContent{Content: lo.ToPtr("Dynamic reminder")}},
		{Role: "assistant", Content: MessageContent{Content: lo.ToPtr("Response")}},
		{Role: "system", Content: MessageContent{Content: lo.ToPtr("Another reminder")}},
	}}

	stabilizeCachePrefix(req)

	require.Len(t, req.Messages, 6)
	require.Equal(t, []string{"system", "system", "user", "user", "assistant", "user"}, []string{
		req.Messages[0].Role,
		req.Messages[1].Role,
		req.Messages[2].Role,
		req.Messages[3].Role,
		req.Messages[4].Role,
		req.Messages[5].Role,
	})
	require.Equal(
		t,
		"x-anthropic-billing-header: cc_version=2.1.177.c0b; You are Claude Code.",
		lo.FromPtr(req.Messages[0].Content.Content),
	)
	require.Equal(t, "Stable system prompt", lo.FromPtr(req.Messages[1].Content.Content))
	require.Equal(t, "Dynamic reminder", lo.FromPtr(req.Messages[3].Content.Content))
}

func TestStabilizeCachePrefix_MultipleContent(t *testing.T) {
	req := &Request{Messages: []Message{{
		Role: "system",
		Content: MessageContent{MultipleContent: []MessageContentPart{
			{Type: "text", Text: lo.ToPtr("x-anthropic-billing-header: cc_version=2.1; cch=abc12;")},
			{Type: "text", Text: lo.ToPtr("Stable prompt; cch=def34; tail")},
		}},
	}}}

	stabilizeCachePrefix(req)

	require.Len(t, req.Messages, 1)
	require.Len(t, req.Messages[0].Content.MultipleContent, 1)
	require.Equal(t, "Stable prompt; tail", lo.FromPtr(req.Messages[0].Content.MultipleContent[0].Text))
}
