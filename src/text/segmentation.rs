use lazy_static::lazy_static;
use std::collections::HashSet;

lazy_static! {
    static ref PUNCTUATION_SET: HashSet<char> = {
        let mut s = HashSet::new();
        s.insert('!');
        s.insert('?');
        s.insert('…');
        s.insert(',');
        s.insert('.');
        s.insert('-');
        s.insert(' ');
        s.insert('，');
        s.insert('。');
        s.insert('？');
        s.insert('！');
        s.insert('、');
        s.insert('；');
        s.insert('：');
        s.insert('~');
        s.insert('—');
        s.insert(':');
        s.insert(';');
        s
    };
}

/// Check if text is only punctuation/whitespace
fn is_pure_punctuation(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    trimmed.chars().all(|c| PUNCTUATION_SET.contains(&c))
}

/// Helper to split text by punctuation boundaries into small clauses while retaining delimiter
fn split_by_clause(todo_text: &str) -> Vec<String> {
    let mut text = todo_text.replace("……", "。").replace("——", "，");
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    if let Some(&last) = chars.last() {
        if !PUNCTUATION_SET.contains(&last) {
            text.push('。');
        }
    }

    let mut result = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if PUNCTUATION_SET.contains(&ch) {
            result.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}

/// Safety fallback: split any big text exceeding max_len (510 chars) by nearest punctuation
pub fn split_big_text(text: &str, max_len: usize) -> Vec<String> {
    if text.chars().count() <= max_len {
        return vec![text.to_string()];
    }

    let clauses = split_by_clause(text);
    let mut result = Vec::new();
    let mut current = String::new();

    for clause in clauses {
        let current_len = current.chars().count();
        let clause_len = clause.chars().count();

        if current_len + clause_len > max_len && current_len > 0 {
            result.push(std::mem::take(&mut current));
            current = clause;
        } else {
            current.push_str(&clause);
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    if result.is_empty() {
        vec![text.to_string()]
    } else {
        result
    }
}

/// cut0: No splitting
pub fn cut0(inp: &str) -> Vec<String> {
    let trimmed = inp.trim().to_string();
    if is_pure_punctuation(&trimmed) {
        Vec::new()
    } else {
        vec![trimmed]
    }
}

/// cut1: Four-clause batching
pub fn cut1(inp: &str) -> Vec<String> {
    let inps = split_by_clause(inp.trim());
    if inps.len() <= 4 {
        let joined = inps.concat();
        return if is_pure_punctuation(&joined) {
            Vec::new()
        } else {
            vec![joined]
        };
    }

    let mut opts = Vec::new();
    for chunk in inps.chunks(4) {
        let joined = chunk.concat();
        if !is_pure_punctuation(&joined) {
            opts.push(joined);
        }
    }
    opts
}

/// cut2: 50-character threshold batching
pub fn cut2(inp: &str) -> Vec<String> {
    let inps = split_by_clause(inp.trim());
    if inps.len() < 2 {
        let joined = inps.concat();
        return if is_pure_punctuation(&joined) {
            Vec::new()
        } else {
            vec![joined]
        };
    }

    let mut opts = Vec::new();
    let mut current = String::new();

    for clause in inps {
        current.push_str(&clause);
        if current.chars().count() >= 50 {
            opts.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        opts.push(current);
    }

    // Merge trailing chunk if too short (< 50 chars) and previous chunk exists
    if opts.len() > 1 {
        let last_len = opts.last().map(|s| s.chars().count()).unwrap_or(0);
        if last_len < 50 {
            let last = opts.pop().unwrap();
            if let Some(prev) = opts.last_mut() {
                prev.push_str(&last);
            } else {
                opts.push(last);
            }
        }
    }

    opts.into_iter().filter(|s| !is_pure_punctuation(s)).collect()
}

/// cut3: Split on Chinese period `。`
pub fn cut3(inp: &str) -> Vec<String> {
    let trimmed = inp.trim();
    let chars: Vec<char> = trimmed.chars().collect();
    let mut result = Vec::new();
    let mut current = String::new();

    for &ch in &chars {
        current.push(ch);
        if ch == '。' {
            let seg = std::mem::take(&mut current);
            if !is_pure_punctuation(&seg) {
                result.push(seg);
            }
        }
    }

    if !current.is_empty() && !is_pure_punctuation(&current) {
        result.push(current);
    }

    result
}

/// cut4: Split on English period `.` (respecting decimals like 3.14)
pub fn cut4(inp: &str) -> Vec<String> {
    let trimmed = inp.trim();
    let chars: Vec<char> = trimmed.chars().collect();
    let mut result = Vec::new();
    let mut current = String::new();

    for i in 0..chars.len() {
        let ch = chars[i];
        current.push(ch);

        if ch == '.' {
            let is_decimal = i > 0
                && i + 1 < chars.len()
                && chars[i - 1].is_ascii_digit()
                && chars[i + 1].is_ascii_digit();

            if !is_decimal {
                let seg = std::mem::take(&mut current);
                if !is_pure_punctuation(&seg) {
                    result.push(seg);
                }
            }
        }
    }

    if !current.is_empty() && !is_pure_punctuation(&current) {
        result.push(current);
    }

    result
}

/// cut5: Split on all punctuation delimiters
pub fn cut5(inp: &str) -> Vec<String> {
    let trimmed = inp.trim();
    let chars: Vec<char> = trimmed.chars().collect();
    let mut merge_items = Vec::new();
    let mut current = String::new();

    for i in 0..chars.len() {
        let ch = chars[i];
        current.push(ch);

        if PUNCTUATION_SET.contains(&ch) {
            // Ignore decimal dot like 3.14
            let is_decimal = ch == '.'
                && i > 0
                && i + 1 < chars.len()
                && chars[i - 1].is_ascii_digit()
                && chars[i + 1].is_ascii_digit();

            if !is_decimal {
                merge_items.push(std::mem::take(&mut current));
            }
        }
    }

    if !current.is_empty() {
        merge_items.push(current);
    }

    merge_items
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !is_pure_punctuation(s))
        .collect()
}

/// Master text segmenter dispatching to cut0~cut5 with max_len safety boundary check
pub fn segment_text(text: &str, method: &str) -> Vec<String> {
    let clean = text.trim();
    if clean.is_empty() {
        return Vec::new();
    }

    let initial_segments = match method.to_lowercase().as_str() {
        "cut0" => cut0(clean),
        "cut1" => cut1(clean),
        "cut2" => cut2(clean),
        "cut3" => cut3(clean),
        "cut4" => cut4(clean),
        "cut5" => cut5(clean),
        _ => cut5(clean),
    };

    // Apply safety split_big_text (> 510 chars) on each chunk
    let mut final_segments = Vec::new();
    for seg in initial_segments {
        if seg.chars().count() > 510 {
            final_segments.extend(split_big_text(&seg, 500));
        } else if !is_pure_punctuation(&seg) {
            final_segments.push(seg);
        }
    }

    if final_segments.is_empty() && !is_pure_punctuation(clean) {
        vec![clean.to_string()]
    } else {
        final_segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut0() {
        let res = cut0("你好，世界！今天天氣真好。");
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_cut3_chinese_period() {
        let res = cut3("第一句話。第二句話。第三句話。");
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn test_cut4_english_period() {
        let res = cut4("Version 3.14 is released. Please test it. All good.");
        assert_eq!(res.len(), 3);
        assert!(res[0].contains("3.14"));
    }

    #[test]
    fn test_cut5_all_punctuations() {
        let res = cut5("你好，世界！今天天氣真好。");
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], "你好，");
        assert_eq!(res[1], "世界！");
        assert_eq!(res[2], "今天天氣真好。");
    }

    #[test]
    fn test_segment_text_dispatch() {
        let res = segment_text("你好，世界！", "cut5");
        assert_eq!(res.len(), 2);

        let res_cut0 = segment_text("你好，世界！", "cut0");
        assert_eq!(res_cut0.len(), 1);
    }
}
