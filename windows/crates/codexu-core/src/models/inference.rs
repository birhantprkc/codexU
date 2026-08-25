use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};

pub const INFERENCE_MINIMUM_CALL_DURATION_SECONDS: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InferencePerformancePeriod {
    Today,
    SevenDays,
    TwentyEightDays,
}

impl InferencePerformancePeriod {
    pub fn day_count(&self) -> i64 {
        match self {
            InferencePerformancePeriod::Today => 1,
            InferencePerformancePeriod::SevenDays => 7,
            InferencePerformancePeriod::TwentyEightDays => 28,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferencePerformanceSample {
    pub sample_id: String,
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub completed_at: DateTime<Utc>,
    pub duration_seconds: f64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
    pub model: String,
    pub effort: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferencePerformanceGroup {
    pub id: String,
    pub model: String,
    pub effort: String,
    pub call_count: i64,
    pub average_daily_call_count: f64,
    pub average_duration_seconds: f64,
    pub p50_duration_seconds: f64,
    pub p90_duration_seconds: f64,
    pub effective_output_tokens_per_second: f64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferencePerformance {
    pub period: InferencePerformancePeriod,
    pub coverage_day_count: i64,
    pub groups: Vec<InferencePerformanceGroup>,
    pub total_call_count: i64,
}

impl InferencePerformance {
    pub fn display_groups(&self) -> Vec<InferencePerformanceGroup> {
        let mut groups = self.groups.clone();
        groups.sort_by(|a, b| {
            b.call_count
                .cmp(&a.call_count)
                .then_with(|| {
                    a.p50_duration_seconds
                        .partial_cmp(&b.p50_duration_seconds)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.id.cmp(&b.id))
        });
        groups
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferencePerformanceHistory {
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub recording_started_at: DateTime<Utc>,
    pub today: Option<InferencePerformance>,
    pub seven_days: Option<InferencePerformance>,
    pub twenty_eight_days: Option<InferencePerformance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferencePerformanceArchive {
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub recording_started_at: DateTime<Utc>,
    pub samples_by_source_id: HashMap<String, Vec<InferencePerformanceSample>>,
}

impl InferencePerformanceArchive {
    pub fn new(recording_started_at: DateTime<Utc>) -> Self {
        Self {
            recording_started_at,
            samples_by_source_id: HashMap::new(),
        }
    }

    pub fn samples(&self) -> Vec<InferencePerformanceSample> {
        let mut samples: Vec<_> = self
            .samples_by_source_id
            .values()
            .flat_map(|samples| samples.iter().cloned())
            .collect();
        samples.sort_by_key(|sample| sample.completed_at);
        samples
    }

    pub fn replace_samples(
        &mut self,
        source_id: impl AsRef<str>,
        samples: Vec<InferencePerformanceSample>,
        retention_start: DateTime<Utc>,
    ) {
        let source_id = source_id.as_ref().trim();
        if source_id.is_empty() {
            return;
        }

        let mut seen = HashSet::new();
        let mut retained: Vec<_> = samples
            .into_iter()
            .filter(|sample| {
                sample.completed_at >= retention_start
                    && sample.duration_seconds >= INFERENCE_MINIMUM_CALL_DURATION_SECONDS
                    && sample.output_tokens > 0
                    && !sample.model.trim().is_empty()
                    && !sample.effort.trim().is_empty()
                    && seen.insert(sample.sample_id.clone())
            })
            .collect();
        retained.sort_by_key(|sample| sample.completed_at);

        if retained.is_empty() {
            self.samples_by_source_id.remove(source_id);
            return;
        }

        if let Some(earliest) = retained.first().map(|sample| sample.completed_at) {
            if earliest < self.recording_started_at {
                self.recording_started_at = earliest;
            }
        }
        self.samples_by_source_id
            .insert(source_id.to_string(), retained);
    }

    pub fn compact(&mut self, retention_start: DateTime<Utc>, maximum_sample_count: usize) {
        let mut retained: Vec<_> = self
            .samples_by_source_id
            .iter()
            .flat_map(|(source_id, samples)| {
                samples
                    .iter()
                    .filter(move |sample| {
                        sample.completed_at >= retention_start
                            && sample.duration_seconds >= INFERENCE_MINIMUM_CALL_DURATION_SECONDS
                            && sample.output_tokens > 0
                    })
                    .map(move |sample| (source_id.clone(), sample.clone()))
            })
            .collect();

        retained.sort_by(|a, b| {
            b.1.completed_at
                .cmp(&a.1.completed_at)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.sample_id.cmp(&b.1.sample_id))
        });
        retained.truncate(maximum_sample_count);

        let mut grouped: HashMap<String, Vec<InferencePerformanceSample>> = HashMap::new();
        for (source_id, sample) in retained {
            grouped.entry(source_id).or_default().push(sample);
        }
        for samples in grouped.values_mut() {
            samples.sort_by_key(|sample| sample.completed_at);
        }
        self.samples_by_source_id = grouped;
        if let Some(earliest) = self.samples().first().map(|sample| sample.completed_at) {
            self.recording_started_at = earliest;
        }
    }
}

pub struct InferencePerformanceBuilder;

impl InferencePerformanceBuilder {
    pub fn make_history(
        samples: &[InferencePerformanceSample],
        recording_started_at: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> Option<InferencePerformanceHistory> {
        let day_start = Utc
            .with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0)
            .unwrap();
        let day_end = day_start + chrono::Duration::days(1);

        let period = |period: InferencePerformancePeriod| {
            let window_start = day_start - chrono::Duration::days(period.day_count() - 1);
            let recording_day_start = Utc
                .with_ymd_and_hms(
                    recording_started_at.year(),
                    recording_started_at.month(),
                    recording_started_at.day(),
                    0,
                    0,
                    0,
                )
                .unwrap();
            let coverage_start = window_start.max(recording_day_start);
            let elapsed_days = day_start
                .signed_duration_since(coverage_start)
                .num_days()
                .max(0);
            let coverage_day_count = (elapsed_days + 1).clamp(1, period.day_count());
            Self::make(samples, period, window_start, day_end, coverage_day_count)
        };

        let history = InferencePerformanceHistory {
            recording_started_at,
            today: period(InferencePerformancePeriod::Today),
            seven_days: period(InferencePerformancePeriod::SevenDays),
            twenty_eight_days: period(InferencePerformancePeriod::TwentyEightDays),
        };
        if history.today.is_none()
            && history.seven_days.is_none()
            && history.twenty_eight_days.is_none()
        {
            None
        } else {
            Some(history)
        }
    }

    pub fn make(
        samples: &[InferencePerformanceSample],
        period: InferencePerformancePeriod,
        window_start: DateTime<Utc>,
        window_end: DateTime<Utc>,
        coverage_day_count: i64,
    ) -> Option<InferencePerformance> {
        let selected: Vec<_> = samples
            .iter()
            .filter(|sample| {
                sample.completed_at >= window_start
                    && sample.completed_at < window_end
                    && sample.duration_seconds >= INFERENCE_MINIMUM_CALL_DURATION_SECONDS
                    && sample.output_tokens > 0
            })
            .cloned()
            .collect();
        if selected.is_empty() {
            return None;
        }

        let mut grouped: HashMap<String, Vec<InferencePerformanceSample>> = HashMap::new();
        for sample in selected {
            grouped
                .entry(inference_group_id(&sample.model, &sample.effort))
                .or_default()
                .push(sample);
        }

        let mut groups = Vec::new();
        for (id, values) in grouped {
            let Some(first) = values.first() else {
                continue;
            };
            let mut durations: Vec<f64> = values
                .iter()
                .map(|sample| sample.duration_seconds)
                .collect();
            durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let total_duration: f64 = durations.iter().sum();
            let output_tokens: i64 = values.iter().map(|sample| sample.output_tokens).sum();
            if total_duration <= 0.0 || output_tokens <= 0 {
                continue;
            }
            let reasoning_output_tokens: i64 = values
                .iter()
                .map(|sample| sample.reasoning_output_tokens)
                .sum();

            groups.push(InferencePerformanceGroup {
                id,
                model: first.model.clone(),
                effort: first.effort.clone(),
                call_count: values.len() as i64,
                average_daily_call_count: values.len() as f64 / coverage_day_count.max(1) as f64,
                average_duration_seconds: total_duration / values.len() as f64,
                p50_duration_seconds: percentile(&durations, 0.5),
                p90_duration_seconds: percentile(&durations, 0.9),
                effective_output_tokens_per_second: output_tokens as f64 / total_duration,
                output_tokens,
                reasoning_output_tokens,
            });
        }

        if groups.is_empty() {
            return None;
        }
        groups.sort_by(|a, b| {
            b.call_count
                .cmp(&a.call_count)
                .then_with(|| a.id.cmp(&b.id))
        });
        let total_call_count = groups.iter().map(|group| group.call_count).sum();
        Some(InferencePerformance {
            period,
            coverage_day_count: coverage_day_count.max(1),
            groups,
            total_call_count,
        })
    }
}

fn percentile(sorted_values: &[f64], fraction: f64) -> f64 {
    let Some(first) = sorted_values.first() else {
        return 0.0;
    };
    if sorted_values.len() == 1 {
        return *first;
    }
    let clamped = fraction.clamp(0.0, 1.0);
    let position = (sorted_values.len() - 1) as f64 * clamped;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted_values[lower];
    }
    let progress = position - lower as f64;
    sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * progress
}

pub fn inference_group_id(model: &str, effort: &str) -> String {
    format!("{}::{}", model.to_lowercase(), effort.to_lowercase())
}
