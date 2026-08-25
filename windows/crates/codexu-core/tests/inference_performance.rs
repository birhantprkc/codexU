use chrono::{Duration, TimeZone, Utc};
use codexu_core::{
    readers::{CodexDashboardProvider, InferencePerformanceReader},
    InferencePerformanceArchive, InferencePerformancePeriod, InferencePerformanceSample,
};
use tempfile::tempdir;

fn write_session(path: &std::path::Path, lines: Vec<String>) {
    std::fs::write(path, lines.join("\n")).unwrap();
}

fn line(
    timestamp: chrono::DateTime<Utc>,
    envelope_type: &str,
    payload: serde_json::Value,
) -> String {
    serde_json::json!({
        "timestamp": timestamp.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        "type": envelope_type,
        "payload": payload,
    })
    .to_string()
}

fn turn_context(
    timestamp: chrono::DateTime<Utc>,
    turn_id: &str,
    model: &str,
    effort: Option<&str>,
) -> String {
    let mut payload = serde_json::json!({
        "turn_id": turn_id,
        "cwd": "C:\\Projects\\Inference",
        "model": model,
    });
    if let Some(effort) = effort {
        payload["effort"] = serde_json::json!(effort);
    }
    line(timestamp, "turn_context", payload)
}

fn assistant_output(timestamp: chrono::DateTime<Utc>) -> String {
    line(
        timestamp,
        "response_item",
        serde_json::json!({
            "type": "agent_message",
            "role": "assistant"
        }),
    )
}

fn tool_output_boundary(timestamp: chrono::DateTime<Utc>) -> String {
    line(
        timestamp,
        "response_item",
        serde_json::json!({
            "type": "function_call_output",
        }),
    )
}

fn token_count(
    timestamp: chrono::DateTime<Utc>,
    turn_id: &str,
    output_tokens: i64,
    reasoning_output_tokens: i64,
) -> String {
    line(
        timestamp,
        "event_msg",
        serde_json::json!({
            "type": "token_count",
            "turn_id": turn_id,
            "info": {
                "last_token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": reasoning_output_tokens,
                    "total_tokens": 100 + output_tokens
                },
                "total_token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": reasoning_output_tokens,
                    "total_tokens": 100 + output_tokens
                }
            }
        }),
    )
}

fn token_count_with_info(
    timestamp: chrono::DateTime<Utc>,
    turn_id: &str,
    info: serde_json::Value,
) -> String {
    line(
        timestamp,
        "event_msg",
        serde_json::json!({
            "type": "token_count",
            "turn_id": turn_id,
            "info": info,
        }),
    )
}

fn inference_history(
    snapshot: &codexu_core::CodexDashboardSnapshot,
) -> &codexu_core::InferencePerformanceHistory {
    snapshot
        .codex
        .snapshot
        .local
        .as_ref()
        .expect("local usage remains present")
        .inference_performance
        .as_ref()
        .expect("inference performance is an independent local branch")
}

#[tokio::test]
async fn dashboard_builds_privacy_safe_inference_performance_branch() {
    let temp = tempdir().unwrap();
    let archived = temp.path().join("archived_sessions");
    std::fs::create_dir_all(&archived).unwrap();

    let today = Utc.with_ymd_and_hms(2026, 8, 5, 12, 0, 0).unwrap();
    let prior_day = today - Duration::days(1);
    let session = archived.join("rollout-inference.jsonl");

    write_session(
        &session,
        vec![
            line(
                prior_day - Duration::seconds(10),
                "session_meta",
                serde_json::json!({"id": "thread-inference", "cwd": "C:\\Projects\\Inference"}),
            ),
            turn_context(prior_day, "turn-prior", "gpt-5", Some("High")),
            assistant_output(prior_day + Duration::seconds(1)),
            token_count(prior_day + Duration::seconds(8), "turn-prior", 80, 20),
            turn_context(today, "turn-1", "gpt-5", Some("High")),
            assistant_output(today + Duration::seconds(1)),
            token_count(today + Duration::seconds(2), "turn-1", 20, 5),
            tool_output_boundary(today + Duration::seconds(10)),
            assistant_output(today + Duration::seconds(13)),
            token_count(today + Duration::seconds(14), "turn-2", 40, 60),
            token_count(today + Duration::milliseconds(14_500), "turn-2", 40, 60),
            turn_context(
                today + Duration::seconds(20),
                "turn-missing-effort",
                "gpt-5",
                None,
            ),
            assistant_output(today + Duration::seconds(21)),
            token_count(today + Duration::seconds(22), "turn-missing-effort", 10, 2),
            turn_context(
                today + Duration::seconds(30),
                "turn-noise",
                "gpt-5",
                Some("high"),
            ),
            assistant_output(today + Duration::milliseconds(30_020)),
            token_count(today + Duration::milliseconds(30_050), "turn-noise", 10, 2),
            turn_context(
                today + Duration::seconds(40),
                "turn-no-output",
                "gpt-5",
                Some("high"),
            ),
            assistant_output(today + Duration::seconds(41)),
            token_count(today + Duration::seconds(42), "turn-no-output", 0, 0),
            turn_context(
                today + Duration::seconds(50),
                "turn-no-model-output",
                "gpt-5",
                Some("high"),
            ),
            token_count(today + Duration::seconds(52), "turn-no-model-output", 10, 2),
        ],
    );

    let provider = CodexDashboardProvider::new(temp.path(), temp.path().join("cache"));
    let snapshot = provider
        .load_dashboard_snapshot(today)
        .await
        .unwrap()
        .expect("snapshot");

    let local = snapshot
        .codex
        .snapshot
        .local
        .as_ref()
        .expect("local usage remains present");
    let inference = local
        .inference_performance
        .as_ref()
        .expect("inference performance is an independent local branch");
    assert!(
        local.detailed_usage.is_some(),
        "existing detailed usage remains usable"
    );

    let today_period = inference.today.as_ref().expect("today period");
    assert_eq!(today_period.period, InferencePerformancePeriod::Today);
    assert_eq!(today_period.total_call_count, 2);
    let today_group = today_period.groups.first().expect("today group");
    assert_eq!(today_group.model, "gpt-5");
    assert_eq!(today_group.effort, "high");
    assert_eq!(today_group.call_count, 2);
    assert_eq!(today_group.output_tokens, 60);
    assert_eq!(
        today_group.reasoning_output_tokens, 45,
        "reasoning tokens are clamped to output-token bounds"
    );
    assert!(
        (today_group.average_duration_seconds - 3.0).abs() < 0.000_1,
        "tool-output boundary excludes tool execution time"
    );

    let seven_days = inference.seven_days.as_ref().expect("seven-day period");
    assert_eq!(seven_days.coverage_day_count, 2);
    assert_eq!(
        seven_days.total_call_count, 3,
        "duplicate and invalid samples are excluded"
    );
    let group = seven_days.groups.first().expect("seven-day group");
    assert!((group.average_daily_call_count - 1.5).abs() < 0.000_1);
    assert!((group.p50_duration_seconds - 4.0).abs() < 0.000_1);
    assert!((group.p90_duration_seconds - 7.2).abs() < 0.000_1);
    assert!(
        (group.effective_output_tokens_per_second - 10.0).abs() < 0.000_1,
        "effective throughput is total output tokens over full call duration"
    );

    let json = serde_json::to_value(&snapshot).unwrap();
    assert!(json["codex"]["snapshot"]["local"]["inference_performance"].is_object());
    assert!(
        json["codex"]["snapshot"]["local"]["detailed_usage"]["inference_performance"].is_null(),
        "inference fields must not be overloaded into DetailedUsage"
    );
}

#[tokio::test]
async fn keeps_multiple_valid_model_calls_inside_one_turn() {
    let temp = tempdir().unwrap();
    let archived = temp.path().join("archived_sessions");
    std::fs::create_dir_all(&archived).unwrap();

    let now = Utc.with_ymd_and_hms(2026, 8, 5, 15, 0, 0).unwrap();
    let session = archived.join("rollout-multi-call.jsonl");
    write_session(
        &session,
        vec![
            line(
                now - Duration::seconds(10),
                "session_meta",
                serde_json::json!({"id": "thread-multi-call", "cwd": "C:\\Projects\\Inference"}),
            ),
            turn_context(now, "turn-shared", "gpt-5", Some("medium")),
            assistant_output(now + Duration::seconds(1)),
            token_count(now + Duration::seconds(3), "turn-shared", 30, 3),
            tool_output_boundary(now + Duration::seconds(10)),
            assistant_output(now + Duration::seconds(11)),
            token_count(now + Duration::seconds(15), "turn-shared", 80, 20),
        ],
    );

    let provider = CodexDashboardProvider::new(temp.path(), temp.path().join("cache"));
    let snapshot = provider
        .load_dashboard_snapshot(now)
        .await
        .unwrap()
        .expect("snapshot");

    let today = inference_history(&snapshot).today.as_ref().expect("today");
    assert_eq!(
        today.total_call_count, 2,
        "two valid model calls in one turn must not collapse to one sample"
    );
    let group = today.groups.first().expect("group");
    assert_eq!(group.call_count, 2);
    assert_eq!(group.output_tokens, 110);
    assert!((group.average_duration_seconds - 4.0).abs() < 0.000_1);
    assert!((group.p50_duration_seconds - 4.0).abs() < 0.000_1);
    assert!((group.effective_output_tokens_per_second - (110.0 / 8.0)).abs() < 0.000_1);
}

#[tokio::test]
async fn copied_rollout_between_live_and_archive_dirs_is_reconciled_once() {
    let temp = tempdir().unwrap();
    let archived = temp.path().join("archived_sessions");
    let sessions = temp.path().join("sessions");
    std::fs::create_dir_all(&archived).unwrap();
    std::fs::create_dir_all(&sessions).unwrap();

    let now = Utc.with_ymd_and_hms(2026, 8, 5, 16, 0, 0).unwrap();
    let lines = vec![
        line(
            now - Duration::seconds(10),
            "session_meta",
            serde_json::json!({"id": "stable-thread-id", "cwd": "C:\\Projects\\Inference"}),
        ),
        turn_context(now, "turn-copy", "gpt-5", Some("high")),
        assistant_output(now + Duration::seconds(1)),
        token_count(now + Duration::seconds(4), "turn-copy", 40, 10),
    ];
    write_session(&archived.join("rollout-copy.jsonl"), lines.clone());
    write_session(&sessions.join("rollout-copy.jsonl"), lines);

    let provider = CodexDashboardProvider::new(temp.path(), temp.path().join("cache"));
    let snapshot = provider
        .load_dashboard_snapshot(now)
        .await
        .unwrap()
        .expect("snapshot");

    let today = inference_history(&snapshot).today.as_ref().expect("today");
    assert_eq!(
        today.total_call_count, 1,
        "the same rollout observed in sessions and archived_sessions should not double-count"
    );
}

#[tokio::test]
async fn merges_partial_live_and_archive_files_for_same_session() {
    let temp = tempdir().unwrap();
    let archived = temp.path().join("archived_sessions");
    let sessions = temp.path().join("sessions");
    std::fs::create_dir_all(&archived).unwrap();
    std::fs::create_dir_all(&sessions).unwrap();

    let now = Utc.with_ymd_and_hms(2026, 8, 5, 16, 30, 0).unwrap();
    let session_meta = line(
        now - Duration::seconds(10),
        "session_meta",
        serde_json::json!({"id": "stable-merge-thread", "cwd": "C:\\Projects\\Inference"}),
    );
    let archive_sample = vec![
        turn_context(now, "turn-archive", "gpt-5", Some("high")),
        assistant_output(now + Duration::seconds(1)),
        token_count(now + Duration::seconds(4), "turn-archive", 40, 10),
    ];
    let shared_live_sample = vec![
        turn_context(
            now + Duration::seconds(5),
            "turn-live",
            "gpt-5",
            Some("high"),
        ),
        assistant_output(now + Duration::seconds(6)),
        token_count(now + Duration::seconds(9), "turn-live", 60, 20),
    ];

    let mut archive_lines = vec![session_meta.clone()];
    archive_lines.extend(archive_sample);
    archive_lines.extend(shared_live_sample.clone());
    write_session(&archived.join("rollout-merge.jsonl"), archive_lines);

    let mut live_lines = vec![session_meta];
    live_lines.extend(shared_live_sample);
    write_session(&sessions.join("rollout-merge.jsonl"), live_lines);

    let provider = CodexDashboardProvider::new(temp.path(), temp.path().join("cache"));
    let snapshot = provider
        .load_dashboard_snapshot(now)
        .await
        .unwrap()
        .expect("snapshot");

    let today = inference_history(&snapshot).today.as_ref().expect("today");
    assert_eq!(
        today.total_call_count, 2,
        "partial live data must merge with archive data without dropping or duplicating samples"
    );
    let group = today.groups.first().expect("group");
    assert_eq!(group.call_count, 2);
    assert_eq!(group.output_tokens, 100);
}

#[tokio::test]
async fn archive_disk_load_save_and_safe_degradation() {
    let temp = tempdir().unwrap();
    let cache = temp.path().join("cache");
    let cache_file = cache.join("codex").join("inference-performance-v1.json");
    std::fs::create_dir_all(cache_file.parent().unwrap()).unwrap();

    let now = Utc.with_ymd_and_hms(2026, 8, 5, 17, 0, 0).unwrap();
    let sample = serde_json::json!({
        "sample_id": "disk-sample",
        "completed_at": (now + Duration::seconds(2)).timestamp_millis(),
        "duration_seconds": 2.0,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "model": "gpt-5",
        "effort": "high",
    });
    let valid_cache = serde_json::json!({
        "version": 1,
        "archive": {
            "recording_started_at": now.timestamp_millis(),
            "samples_by_source_id": {
                "stable-thread-id": [sample],
            }
        }
    });
    std::fs::write(&cache_file, serde_json::to_vec(&valid_cache).unwrap()).unwrap();

    let reader = InferencePerformanceReader::new(&cache);
    let loaded = reader
        .load(temp.path(), now)
        .await
        .unwrap()
        .expect("loaded history");
    assert_eq!(loaded.today.as_ref().unwrap().total_call_count, 1);
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&cache_file).unwrap()).unwrap();
    assert_eq!(saved["version"], 1);

    std::fs::write(&cache_file, b"not-json").unwrap();
    assert!(
        reader.load(temp.path(), now).await.unwrap().is_none(),
        "corrupt archive must safely degrade to empty data"
    );

    let version_mismatch = serde_json::json!({
        "version": 999,
        "archive": {
            "recording_started_at": now.timestamp_millis(),
            "samples_by_source_id": {
                "stable-thread-id": [sample],
            }
        }
    });
    std::fs::write(&cache_file, serde_json::to_vec(&version_mismatch).unwrap()).unwrap();
    assert!(
        reader.load(temp.path(), now).await.unwrap().is_none(),
        "version mismatch must safely degrade to empty data"
    );

    std::fs::write(&cache_file, vec![b'x'; 33 * 1024 * 1024]).unwrap();
    assert!(
        reader.load(temp.path(), now).await.unwrap().is_none(),
        "oversized archive must safely degrade to empty data"
    );
}

#[tokio::test]
async fn ignores_token_events_with_missing_info_last_usage_or_required_numeric_fields() {
    let temp = tempdir().unwrap();
    let archived = temp.path().join("archived_sessions");
    std::fs::create_dir_all(&archived).unwrap();

    let now = Utc.with_ymd_and_hms(2026, 8, 5, 18, 0, 0).unwrap();
    let session = archived.join("rollout-invalid-token-events.jsonl");
    write_session(
        &session,
        vec![
            line(
                now - Duration::seconds(10),
                "session_meta",
                serde_json::json!({"id": "thread-invalid-token-events", "cwd": "C:\\Projects\\Inference"}),
            ),
            turn_context(now, "turn-missing-info", "gpt-5", Some("high")),
            assistant_output(now + Duration::seconds(1)),
            line(
                now + Duration::seconds(3),
                "event_msg",
                serde_json::json!({"type": "token_count", "turn_id": "turn-missing-info"}),
            ),
            turn_context(
                now + Duration::seconds(10),
                "turn-missing-last",
                "gpt-5",
                Some("high"),
            ),
            assistant_output(now + Duration::seconds(11)),
            token_count_with_info(
                now + Duration::seconds(13),
                "turn-missing-last",
                serde_json::json!({
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 0,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120
                    }
                }),
            ),
            turn_context(
                now + Duration::seconds(20),
                "turn-missing-output",
                "gpt-5",
                Some("high"),
            ),
            assistant_output(now + Duration::seconds(21)),
            token_count_with_info(
                now + Duration::seconds(23),
                "turn-missing-output",
                serde_json::json!({
                    "last_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 0,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120
                    }
                }),
            ),
            turn_context(
                now + Duration::seconds(30),
                "turn-valid",
                "gpt-5",
                Some("high"),
            ),
            assistant_output(now + Duration::seconds(31)),
            token_count(now + Duration::seconds(34), "turn-valid", 40, 10),
        ],
    );

    let provider = CodexDashboardProvider::new(temp.path(), temp.path().join("cache"));
    let snapshot = provider
        .load_dashboard_snapshot(now)
        .await
        .unwrap()
        .expect("snapshot");

    let today = inference_history(&snapshot).today.as_ref().expect("today");
    assert_eq!(
        today.total_call_count, 1,
        "only the event with info.last_token_usage and required numeric token fields should be retained"
    );
    assert_eq!(today.groups.first().unwrap().output_tokens, 40);
}

#[test]
fn inference_archive_deduplicates_and_bounds_retention() {
    let base = Utc.with_ymd_and_hms(2026, 8, 5, 12, 0, 0).unwrap();
    let retention_start = base - Duration::days(28);
    let mut archive = InferencePerformanceArchive::new(base);
    let sample = |id: &str, completed_at: chrono::DateTime<Utc>, duration_seconds: f64| {
        InferencePerformanceSample {
            sample_id: id.to_string(),
            completed_at,
            duration_seconds,
            output_tokens: 10,
            reasoning_output_tokens: 2,
            model: "gpt-5".to_string(),
            effort: "high".to_string(),
        }
    };

    archive.replace_samples(
        "rollout-a",
        vec![
            sample("old", retention_start - Duration::seconds(1), 2.0),
            sample("noise", base, 0.05),
            sample("a", base - Duration::seconds(3), 2.0),
            sample("a", base - Duration::seconds(2), 2.0),
            sample("b", base - Duration::seconds(1), 3.0),
        ],
        retention_start,
    );

    assert_eq!(
        archive.samples().len(),
        2,
        "old, noisy, and duplicate samples are discarded"
    );
    archive.compact(retention_start, 1);
    assert_eq!(
        archive.samples().len(),
        1,
        "archive enforces maximum sample count"
    );
    assert_eq!(
        archive.samples()[0].sample_id,
        "b",
        "newest sample is retained first"
    );
}
