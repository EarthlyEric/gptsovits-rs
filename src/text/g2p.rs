use crate::text::cmu_dict::get_english_phonemes;
use crate::text::normalizer::normalize_chinese_text;
use crate::text::pinyin_dict::{OPENCPOP_DICT, PINYIN_DICT};
use lazy_static::lazy_static;
use regex::Regex;
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
        s
    };
    static ref WORD_SPLIT_RE: Regex =
        Regex::new(r"[\u4e00-\u9fa5]|[a-zA-Z]+|[!?,.…\-]|\s+").unwrap();
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PinyinPhone {
    pub initial: String,
    pub final_with_tone: String,
    pub raw_tone: u8,
}

/// Apply tone sandhi modifications for Chinese (3rd tone sandhi, "一", "不")
pub fn apply_tone_sandhi(chars: &[char], pinyins: &mut [String]) {
    let n = chars.len();
    if n < 2 {
        return;
    }

    for i in 0..n {
        let ch = chars[i];
        let py = &pinyins[i];
        if py.is_empty() {
            continue;
        }

        let tone = py.chars().last().and_then(|c| c.to_digit(10)).unwrap_or(5) as u8;
        let base_py = if py.ends_with(|c: char| c.is_ascii_digit()) {
            &py[..py.len() - 1]
        } else {
            py.as_str()
        };

        // Rule for "一" (yi)
        if ch == '一' {
            if i + 1 < n {
                let next_py = &pinyins[i + 1];
                let next_tone = next_py
                    .chars()
                    .last()
                    .and_then(|c| c.to_digit(10))
                    .unwrap_or(5);
                if next_tone == 4 {
                    pinyins[i] = format!("{}2", base_py);
                } else if next_tone == 1 || next_tone == 2 || next_tone == 3 {
                    pinyins[i] = format!("{}4", base_py);
                }
            }
        }
        // Rule for "不" (bu)
        else if ch == '不' {
            if i + 1 < n {
                let next_py = &pinyins[i + 1];
                let next_tone = next_py
                    .chars()
                    .last()
                    .and_then(|c| c.to_digit(10))
                    .unwrap_or(5);
                if next_tone == 4 {
                    pinyins[i] = format!("{}2", base_py);
                }
            }
        }
        // Rule for 3rd tone + 3rd tone -> 2nd tone + 3rd tone
        else if tone == 3 && i + 1 < n {
            let next_py = &pinyins[i + 1];
            let next_tone = next_py
                .chars()
                .last()
                .and_then(|c| c.to_digit(10))
                .unwrap_or(5);
            if next_tone == 3 {
                pinyins[i] = format!("{}2", base_py);
            }
        }
    }
}

/// Convert a single pinyin string (e.g. "hao3", "zheng4") into (initial, final_tone)
pub fn pinyin_to_opencpop(pinyin_with_tone: &str) -> Option<(String, String)> {
    if pinyin_with_tone.is_empty() {
        return None;
    }

    let tone = pinyin_with_tone
        .chars()
        .last()
        .and_then(|c| c.to_digit(10))
        .unwrap_or(5);
    let base = if pinyin_with_tone.ends_with(|c: char| c.is_ascii_digit()) {
        &pinyin_with_tone[..pinyin_with_tone.len() - 1]
    } else {
        pinyin_with_tone
    };

    if let Some(&(c, v)) = OPENCPOP_DICT.get(base) {
        let v_tone = format!("{}{}", v, tone);
        Some((c.to_string(), v_tone))
    } else {
        // Fallback for special or unknown pinyin
        Some(("AA".to_string(), format!("a{}", tone)))
    }
}

/// Convert input text to phoneme sequence, word2ph alignment array, and normalized text
pub fn text_to_phonemes(
    text: &str,
    _language: &str,
    _version: &str,
) -> (Vec<String>, Vec<usize>, String) {
    let norm_text = normalize_chinese_text(text);
    let mut phones = Vec::new();
    let mut word2ph = Vec::new();

    let mut zh_chars = Vec::new();
    let mut zh_indices = Vec::new();

    let chars: Vec<char> = norm_text.chars().collect();

    // First collect all Chinese characters for batch tone sandhi
    for (i, &ch) in chars.iter().enumerate() {
        if ('\u{4e00}'..='\u{9fa5}').contains(&ch) {
            zh_chars.push(ch);
            zh_indices.push(i);
        }
    }

    let mut zh_pinyins = Vec::with_capacity(zh_chars.len());
    for &ch in &zh_chars {
        let ch_str = ch.to_string();
        let py = PINYIN_DICT.get(ch_str.as_str()).copied().unwrap_or("a1");
        zh_pinyins.push(py.to_string());
    }

    apply_tone_sandhi(&zh_chars, &mut zh_pinyins);

    let mut zh_map: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    for (idx, py) in zh_indices.into_iter().zip(zh_pinyins) {
        zh_map.insert(idx, py);
    }

    // Process token by token
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];

        if PUNCTUATION_SET.contains(&ch) {
            let p = ch.to_string();
            phones.push(p);
            word2ph.push(1);
            i += 1;
        } else if ('\u{4e00}'..='\u{9fa5}').contains(&ch) {
            if let Some(py) = zh_map.get(&i) {
                if let Some((c, v)) = pinyin_to_opencpop(py) {
                    phones.push(c);
                    phones.push(v);
                    word2ph.push(2);
                } else {
                    phones.push("AA".to_string());
                    phones.push("a1".to_string());
                    word2ph.push(2);
                }
            } else {
                phones.push("AA".to_string());
                phones.push("a1".to_string());
                word2ph.push(2);
            }
            i += 1;
        } else if ch.is_ascii_alphabetic() {
            // Collect full English word
            let mut word = String::new();
            while i < chars.len() && chars[i].is_ascii_alphabetic() {
                word.push(chars[i]);
                i += 1;
            }
            let word_len = word.chars().count();
            let en_phones = get_english_phonemes(&word);
            let num_en = en_phones.len();
            phones.extend(en_phones);

            // Distribute phone count across characters in the English word
            for char_idx in 0..word_len {
                if char_idx == 0 {
                    word2ph.push(num_en.max(1));
                } else {
                    word2ph.push(0);
                }
            }
        } else {
            // Unsupported symbols are removed by the upstream normalizer. Keep
            // this branch non-phonetic as a final guard instead of inventing a
            // comma phone that changes the prompt alignment.
            word2ph.push(0);
            i += 1;
        }
    }

    // Fallback if empty
    if phones.is_empty() {
        phones.push(".".to_string());
        word2ph.push(1);
    }

    // Ensure at least 4 phonemes as GPT-SoVITS requires
    if phones.len() < 4 {
        phones.insert(0, ",".to_string());
        if !word2ph.is_empty() {
            word2ph[0] += 1;
        } else {
            word2ph.push(1);
        }
    }

    (phones, word2ph, norm_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_phonemes_chinese() {
        let text = "你好，世界！";
        let (phones, word2ph, norm_text) = text_to_phonemes(text, "zh", "v2");
        assert!(!phones.is_empty());
        assert_eq!(norm_text, "你好,世界!");
        assert_eq!(word2ph.len(), norm_text.chars().count());
        assert_eq!(word2ph.iter().sum::<usize>(), phones.len());
    }

    #[test]
    fn test_text_to_phonemes_english() {
        let text = "Hello world!";
        let (phones, _word2ph, _norm_text) = text_to_phonemes(text, "en", "v2");
        assert!(!phones.is_empty());
    }
}
