package openai

import (
	"regexp"
	"strings"
)

var (
	volatileCCHPattern       = regexp.MustCompile(`(?i)\bcch\s*=\s*[^;\s]+`)
	repeatedSemicolonPattern = regexp.MustCompile(`;\s*;`)
)

// stabilizeCachePrefix adapts Claude Code conversations for providers that
// cache an exact prompt prefix instead of honoring Anthropic cache_control.
func stabilizeCachePrefix(req *Request) {
	if req == nil || len(req.Messages) == 0 {
		return
	}

	seenConversation := false
	messages := make([]Message, 0, len(req.Messages))
	for _, msg := range req.Messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role != "system" && role != "developer" {
			seenConversation = true
			messages = append(messages, msg)
			continue
		}

		if role == "system" && seenConversation {
			msg.Role = "user"
			messages = append(messages, msg)
			continue
		}

		if role == "system" && !normalizeLeadingSystemMessage(&msg) {
			continue
		}
		messages = append(messages, msg)
	}

	req.Messages = messages
}

func normalizeLeadingSystemMessage(msg *Message) bool {
	if msg == nil {
		return false
	}

	if msg.Content.Content != nil {
		text := *msg.Content.Content
		if isStandaloneAnthropicBillingHeader(text) {
			return false
		}
		normalized := stripVolatileCCH(text)
		msg.Content.Content = &normalized
		return true
	}

	if len(msg.Content.MultipleContent) == 0 {
		return true
	}

	parts := make([]MessageContentPart, 0, len(msg.Content.MultipleContent))
	for _, part := range msg.Content.MultipleContent {
		if part.Type != "text" || part.Text == nil {
			parts = append(parts, part)
			continue
		}
		if isStandaloneAnthropicBillingHeader(*part.Text) {
			continue
		}
		normalized := stripVolatileCCH(*part.Text)
		part.Text = &normalized
		parts = append(parts, part)
	}
	msg.Content.MultipleContent = parts

	return len(parts) > 0
}

func isStandaloneAnthropicBillingHeader(text string) bool {
	const prefix = "x-anthropic-billing-header:"

	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(strings.ToLower(trimmed), prefix) {
		return false
	}

	foundField := false
	for _, field := range strings.Split(trimmed[len(prefix):], ";") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		key, _, ok := strings.Cut(field, "=")
		if !ok {
			return false
		}
		key = strings.ToLower(strings.TrimSpace(key))
		if key != "cch" && !strings.HasPrefix(key, "cc_") {
			return false
		}
		foundField = true
	}

	return foundField
}

func stripVolatileCCH(text string) string {
	if !volatileCCHPattern.MatchString(text) {
		return text
	}

	result := volatileCCHPattern.ReplaceAllString(text, "")
	for repeatedSemicolonPattern.MatchString(result) {
		result = repeatedSemicolonPattern.ReplaceAllString(result, ";")
	}
	result = strings.TrimSpace(result)
	result = strings.TrimSpace(strings.TrimPrefix(result, ";"))

	return result
}
