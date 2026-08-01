package biz

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/authz"
	"github.com/looplj/axonhub/internal/ent/channel"
	"github.com/looplj/axonhub/internal/ent/enttest"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/codex"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func TestOpenCodeGoChannel_EnablesSessionAffinity(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()

	ctx := authz.WithTestBypass(context.Background())
	entChannel := client.Channel.Create().
		SetName("OpenCode Go Channel").
		SetType(channel.TypeOpencodeGo).
		SetBaseURL("https://opencode.ai/zen/go/v1").
		SetCredentials(objects.ChannelCredentials{APIKey: "test-key"}).
		SetSupportedModels([]string{"deepseek-v4-flash"}).
		SetDefaultTestModel("deepseek-v4-flash").
		SetEndpoints([]objects.ChannelEndpoint{{
			APIFormat: llm.APIFormatOpenAIChatCompletion.String(),
			Path:      "/custom/chat/completions",
		}}).
		SaveX(ctx)

	built, err := NewChannelServiceForTest(client).buildChannelWithOutbounds(entChannel)
	require.NoError(t, err)
	outbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIChatCompletion.String())
	require.NoError(t, err)

	requestCtx := shared.WithSessionID(t.Context(), "opencode-session-123")
	transformed, err := outbound.TransformRequest(requestCtx, &llm.Request{
		Model: "deepseek-v4-flash",
		Messages: []llm.Message{
			{
				Role: "system",
				Content: llm.MessageContent{Content: lo.ToPtr(
					"x-anthropic-billing-header: cc_version=2.1; cch=abc12;",
				)},
			},
			{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("Stable prompt")}},
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("Hello")}},
			{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("Reminder")}},
		},
	})
	require.NoError(t, err)
	require.Equal(t, "opencode-session-123", transformed.Headers.Get("x-session-affinity"))

	var payload openai.Request
	require.NoError(t, json.Unmarshal(transformed.Body, &payload))
	require.Equal(t, "opencode-session-123", lo.FromPtr(payload.PromptCacheKey))
	require.Len(t, payload.Messages, 3)
	require.Equal(t, []string{"system", "user", "user"}, []string{
		payload.Messages[0].Role,
		payload.Messages[1].Role,
		payload.Messages[2].Role,
	})
}

func TestOpenAICompatibleChannel_BuildChannelWithOutbounds(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()

	ctx := authz.WithTestBypass(context.Background())

	entChannel := client.Channel.Create().
		SetName("Vercel Multi Endpoint Channel").
		SetType(channel.TypeVercel).
		SetBaseURL("https://ai-gateway.vercel.sh/v1").
		SetCredentials(objects.ChannelCredentials{APIKey: "test-key"}).
		SetSupportedModels([]string{"gpt-4o-mini"}).
		SetDefaultTestModel("gpt-4o-mini").
		SaveX(ctx)

	channelSvc := NewChannelServiceForTest(client)

	built, err := channelSvc.buildChannelWithOutbounds(entChannel)
	require.NoError(t, err)
	require.NotNil(t, built)
	require.NotNil(t, built.Outbound)
	require.Len(t, built.Outbounds, 7)

	require.Equal(t, llm.APIFormatOpenAIChatCompletion, built.Outbound.APIFormat())

	embeddingOutbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIEmbedding.String())
	require.NoError(t, err)
	require.NotNil(t, embeddingOutbound)
	_, ok := embeddingOutbound.(*openai.OutboundTransformer)
	require.True(t, ok)

	moderationOutbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIModeration.String())
	require.NoError(t, err)
	require.NotNil(t, moderationOutbound)
	_, ok = moderationOutbound.(*openai.OutboundTransformer)
	require.True(t, ok)

	imageOutbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIImageGeneration.String())
	require.NoError(t, err)
	require.NotNil(t, imageOutbound)
	_, ok = imageOutbound.(*openai.OutboundTransformer)
	require.True(t, ok)

	videoOutbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIVideo.String())
	require.NoError(t, err)
	require.NotNil(t, videoOutbound)
	_, ok = videoOutbound.(*openai.OutboundTransformer)
	require.True(t, ok)
}

func TestAtlasCloudChannel_BuildChannelWithOutbounds(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()

	ctx := authz.WithTestBypass(context.Background())

	entChannel := client.Channel.Create().
		SetName("AtlasCloud Channel").
		SetType(channel.TypeAtlascloud).
		SetBaseURL("https://api.atlascloud.ai/v1").
		SetCredentials(objects.ChannelCredentials{APIKey: "test-key"}).
		SetSupportedModels([]string{"deepseek-v3"}).
		SetDefaultTestModel("deepseek-v3").
		SaveX(ctx)

	channelSvc := NewChannelServiceForTest(client)

	built, err := channelSvc.buildChannelWithOutbounds(entChannel)
	require.NoError(t, err)
	require.NotNil(t, built)
	require.NotNil(t, built.Outbound)
	require.Len(t, built.Outbounds, 7)

	require.Equal(t, llm.APIFormatOpenAIChatCompletion, built.Outbound.APIFormat())

	embeddingOutbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIEmbedding.String())
	require.NoError(t, err)
	require.NotNil(t, embeddingOutbound)
	_, ok := embeddingOutbound.(*openai.OutboundTransformer)
	require.True(t, ok)

	moderationOutbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIModeration.String())
	require.NoError(t, err)
	require.NotNil(t, moderationOutbound)
	_, ok = moderationOutbound.(*openai.OutboundTransformer)
	require.True(t, ok)
}

func TestOpenAIResponsesEndpoint_InheritsWebSocketTransportFromBaseURL(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()

	ctx := authz.WithTestBypass(context.Background())

	entChannel := client.Channel.Create().
		SetName("Responses WebSocket Channel").
		SetType(channel.TypeOpenaiResponses).
		SetBaseURL("wss://api.openai.com/v1").
		SetCredentials(objects.ChannelCredentials{APIKey: "test-key"}).
		SetSupportedModels([]string{"gpt-5"}).
		SetDefaultTestModel("gpt-5").
		SetEndpoints([]objects.ChannelEndpoint{{
			APIFormat: llm.APIFormatOpenAIResponse.String(),
			Path:      "/custom/responses",
		}}).
		SaveX(ctx)

	channelSvc := NewChannelServiceForTest(client)

	built, err := channelSvc.buildChannelWithOutbounds(entChannel)
	require.NoError(t, err)

	outbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIResponse.String())
	require.NoError(t, err)
	custom, ok := outbound.(pipeline.ChannelCustomizedExecutor)
	require.True(t, ok)

	executor := custom.CustomizeExecutor(nil)
	_, ok = executor.(*responses.WebSocketExecutor)
	require.True(t, ok)
}

func TestCodexOAuthWebSocketEndpointBuildsWithoutAPIKey(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
	defer client.Close()

	ctx := authz.WithTestBypass(context.Background())

	entChannel := client.Channel.Create().
		SetName("Codex OAuth WebSocket Channel").
		SetType(channel.TypeCodex).
		SetBaseURL("wss://chatgpt.com/backend-api/codex#").
		SetCredentials(objects.ChannelCredentials{
			OAuth: &objects.OAuthCredentials{
				AccessToken:  "access-token",
				RefreshToken: "refresh-token",
				ExpiresAt:    time.Now().Add(time.Hour),
			},
		}).
		SetSupportedModels([]string{"gpt-5.5"}).
		SetDefaultTestModel("gpt-5.5").
		SetEndpoints([]objects.ChannelEndpoint{{
			APIFormat: llm.APIFormatOpenAIResponse.String(),
			Transport: objects.ChannelEndpointTransportWebSocket,
		}}).
		SaveX(ctx)

	channelSvc := NewChannelServiceForTest(client)

	built, err := channelSvc.buildChannelWithOutbounds(entChannel)
	require.NoError(t, err)

	primary, ok := built.Outbound.(*codex.OutboundTransformer)
	require.True(t, ok)
	require.NotNil(t, primary.TokenProvider())

	outbound, err := BuildOutboundByAPIFormat(built, llm.APIFormatOpenAIResponse.String())
	require.NoError(t, err)
	override, ok := outbound.(*codex.OutboundTransformer)
	require.True(t, ok)
	require.True(t, primary.TokenProvider() == override.TokenProvider())

	custom, ok := outbound.(pipeline.ChannelCustomizedExecutor)
	require.True(t, ok)
	require.NotNil(t, custom.CustomizeExecutor(nil))
}

type testStoppableOutbound struct {
	stops int
}

func (t *testStoppableOutbound) APIFormat() llm.APIFormat { return llm.APIFormatOpenAIResponse }

func (t *testStoppableOutbound) TransformRequest(context.Context, *llm.Request) (*httpclient.Request, error) {
	return nil, nil
}

func (t *testStoppableOutbound) TransformResponse(context.Context, *httpclient.Response) (*llm.Response, error) {
	return nil, nil
}

func (t *testStoppableOutbound) TransformStream(context.Context, *httpclient.Request, streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*llm.Response], error) {
	return nil, nil
}

func (t *testStoppableOutbound) TransformError(context.Context, *httpclient.Error) *llm.ResponseError {
	return nil
}

func (t *testStoppableOutbound) AggregateStreamChunks(context.Context, *httpclient.Request, []*httpclient.StreamEvent) ([]byte, llm.ResponseMeta, error) {
	return nil, llm.ResponseMeta{}, nil
}

func (t *testStoppableOutbound) Stop() {
	t.stops++
}

func TestStopChannelOutboundsStopsEachOutboundOnce(t *testing.T) {
	primary := &testStoppableOutbound{}
	secondary := &testStoppableOutbound{}

	stopChannelOutbounds(&Channel{
		Outbound: primary,
		Outbounds: map[string]transformer.Outbound{
			llm.APIFormatOpenAIResponse.String():        primary,
			llm.APIFormatOpenAIResponseCompact.String(): secondary,
		},
	})

	require.Equal(t, 1, primary.stops)
	require.Equal(t, 1, secondary.stops)
}

type closeIdleTrackingTransport struct {
	closeIdleCalls int
}

func (*closeIdleTrackingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, nil
}

func (t *closeIdleTrackingTransport) CloseIdleConnections() {
	t.closeIdleCalls++
}

func TestCleanupSwappedChannelsClosesOnlyOwnedHTTPClients(t *testing.T) {
	sharedTransport := &closeIdleTrackingTransport{}
	ownedTransport := &closeIdleTrackingTransport{}
	sharedClient := httpclient.NewHttpClientWithClient(&http.Client{Transport: sharedTransport})
	ownedClient := httpclient.NewHttpClientWithClient(&http.Client{Transport: ownedTransport})
	svc := &ChannelService{httpClient: sharedClient}

	svc.cleanupSwappedChannels([]*Channel{
		{HTTPClient: sharedClient},
		{HTTPClient: ownedClient},
		{},
	})

	require.Zero(t, sharedTransport.closeIdleCalls)
	require.Equal(t, 1, ownedTransport.closeIdleCalls)
}

func TestOnEnabledChannelsSwapDoesNotWaitForOldCleanup(t *testing.T) {
	cleanupStarted := make(chan struct{})
	releaseCleanup := make(chan struct{})
	cleanupDone := make(chan struct{})
	startCount := 0

	svc := &ChannelService{}
	old := &Channel{
		stopTokenProvider: func() {
			close(cleanupStarted)
			<-releaseCleanup
			close(cleanupDone)
		},
	}
	next := &Channel{
		startTokenProvider: func() {
			startCount++
		},
	}

	returned := make(chan struct{})
	go func() {
		svc.onEnabledChannelsSwap([]*Channel{old}, []*Channel{next})
		close(returned)
	}()

	select {
	case <-returned:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("onEnabledChannelsSwap waited for old channel cleanup")
	}

	require.Equal(t, int64(1), svc.GetCacheVersion())
	require.Equal(t, 1, startCount)

	select {
	case <-cleanupStarted:
	case <-time.After(time.Second):
		t.Fatal("old channel cleanup did not start")
	}

	close(releaseCleanup)

	select {
	case <-cleanupDone:
	case <-time.After(time.Second):
		t.Fatal("old channel cleanup did not finish")
	}
}
