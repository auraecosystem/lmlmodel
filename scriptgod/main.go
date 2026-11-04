package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type Participant struct {
	ID string `json:"id"`
	Endpoint string `json:"endpoint"`
	Capabilities []string `json:"capabilities,omitempty"`
	Connected bool `json:"connected"`
}

type Coordinator struct {
	mu sync.RWMutex
	participants map[string]Participant
	maxBytes int64
}

func NewCoordinator() *Coordinator {
	return &Coordinator{participants: make(map[string]Participant), maxBytes: 1 << 20}
}

func (c *Coordinator) register(p Participant) error {
	if strings.TrimSpace(p.ID) == "" { return errors.New("participant id is required") }
	if p.Endpoint == "" { return errors.New("participant endpoint is required") }
	c.mu.Lock()
	defer c.mu.Unlock()
	p.Connected = true
	c.participants[p.ID] = p
	return nil
}

func (c *Coordinator) participant(id string) (Participant, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	p, ok := c.participants[id]
	return p, ok
}

func (c *Coordinator) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"status":"ok","protocol":ProtocolVersion})
	})
	mux.HandleFunc("GET /god/v1/participants", func(w http.ResponseWriter, r *http.Request) {
		c.mu.RLock()
		defer c.mu.RUnlock()
		out := make([]Participant, 0, len(c.participants))
		for _, p := range c.participants { out = append(out, p) }
		writeJSON(w, http.StatusOK, out)
	})
	mux.HandleFunc("POST /god/v1/register", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, c.maxBytes)
		var p Participant
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			writeError(w, http.StatusBadRequest, err.Error()); return
		}
		if err := c.register(p); err != nil {
			writeError(w, http.StatusBadRequest, err.Error()); return
		}
		writeJSON(w, http.StatusCreated, map[string]any{"registered":true,"participant":p.ID})
	})
	mux.HandleFunc("POST /god/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, c.maxBytes)
		var msg Message
		if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
			writeError(w, http.StatusBadRequest, err.Error()); return
		}
		if err := msg.Validate(); err != nil {
			writeError(w, http.StatusBadRequest, err.Error()); return
		}
		if msg.Sender != "Script.god" {
			if _, ok := c.participant(msg.Sender); !ok {
				writeError(w, http.StatusForbidden, "sender is not registered"); return
			}
		}
		target, ok := c.participant(msg.Recipient)
		if !ok {
			writeError(w, http.StatusNotFound, "recipient is not registered"); return
		}
		if msg.Type == DATA || msg.Type == SYNC || msg.Type == TASK || msg.Type == CONTEXT {
			// Only the coordinator may dispatch privileged coordination messages.
			if msg.Sender != "Script.god" {
				writeError(w, http.StatusForbidden, "only Script.god may dispatch this message type"); return
			}
		}
		msg.Timestamp = time.Now().UTC()
		body, err := json.Marshal(msg)
		if err != nil { writeError(w, http.StatusInternalServerError, err.Error()); return }
		req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, strings.TrimRight(target.Endpoint, "/")+"/god/v1/messages", strings.NewReader(string(body)))
		if err != nil { writeError(w, http.StatusBadGateway, err.Error()); return }
		req.Header.Set("Content-Type", "application/json")
		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(req)
		if err != nil { writeError(w, http.StatusBadGateway, "participant delivery failed: "+err.Error()); return }
		defer resp.Body.Close()
		w.WriteHeader(resp.StatusCode)
		_, _ = fmt.Fprintf(w, "{\"delivered\":%t,\"recipient\":%q,\"status\":%d}\n", resp.StatusCode >= 200 && resp.StatusCode < 300, msg.Recipient, resp.StatusCode)
	})
	return securityHeaders(mux)
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error":message})
}

func main() {
	addr := os.Getenv("GOD_LISTEN")
	if addr == "" { addr = "127.0.0.1:8787" }
	server := &http.Server{
		Addr: addr,
		Handler: NewCoordinator().routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 15 * time.Second,
		WriteTimeout: 35 * time.Second,
		IdleTimeout: 60 * time.Second,
	}
	log.Printf("Script.god runtime %s listening on %s", ProtocolVersion, addr)
	log.Fatal(server.ListenAndServe())
}
