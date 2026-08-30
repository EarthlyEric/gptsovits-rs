use regex::Regex;
use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    static ref REP_MAP: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("：", ",");
        m.insert("；", ",");
        m.insert("，", ",");
        m.insert("。", ".");
        m.insert("！", "!");
        m.insert("？", "?");
        m.insert("\n", ".");
        m.insert("·", ",");
        m.insert("、", ",");
        m.insert("...", "…");
        m.insert("$", ".");
        m.insert("/", ",");
        m.insert("—", "-");
        m.insert("~", "…");
        m.insert("～", "…");
        m
    };

    static ref CONSECUTIVE_PUNCT: Regex = Regex::new(r"([!?,.…\-])([!?,.…\-])+").unwrap();
    static ref DIGITS_RE: Regex = Regex::new(r"\d+").unwrap();
}

/// Verbalize integer number to Chinese characters
pub fn number_to_chinese(num: i64) -> String {
    if num == 0 {
        return "零".to_string();
    }

    let digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
    let units = ["", "十", "百", "千"];
    let big_units = ["", "万", "亿", "兆"];

    let mut n = num;
    let mut negative = false;
    if n < 0 {
        negative = true;
        n = -n;
    }

    let mut result = String::new();
    let mut section_idx = 0;

    while n > 0 {
        let section = (n % 10000) as usize;
        if section > 0 {
            let mut sec_str = String::new();
            let mut zero_flag = false;
            let mut temp = section;

            let mut sec_digits = Vec::new();
            while temp > 0 {
                sec_digits.push(temp % 10);
                temp /= 10;
            }

            for (i, &d) in sec_digits.iter().enumerate() {
                if d == 0 {
                    if !zero_flag && !sec_str.is_empty() {
                        sec_str = format!("零{}", sec_str);
                        zero_flag = true;
                    }
                } else {
                    zero_flag = false;
                    let u = units[i];
                    let d_str = digits[d];
                    // Special case: 10~19 -> "十", "十一", not "一十"
                    if (10..20).contains(&section) && i == 1 && d == 1 {
                        sec_str = format!("{}{}", u, sec_str);
                    } else {
                        sec_str = format!("{}{}{}", d_str, u, sec_str);
                    }
                }
            }

            if !sec_str.is_empty() {
                sec_str = format!("{}{}", sec_str, big_units[section_idx]);
                result = format!("{}{}", sec_str, result);
            }
        }
        n /= 10000;
        section_idx += 1;
    }

    if negative {
        result = format!("负{}", result);
    }

    result
}

/// Convert all digit sequences in text to Chinese numerals
pub fn normalize_numbers(text: &str) -> String {
    DIGITS_RE.replace_all(text, |caps: &regex::Captures| {
        if let Ok(num) = caps[0].parse::<i64>() {
            number_to_chinese(num)
        } else {
            // If overflow or too large, read digit by digit
            let digits_map = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
            caps[0]
                .chars()
                .map(|c| {
                    if let Some(d) = c.to_digit(10) {
                        digits_map[d as usize]
                    } else {
                        ""
                    }
                })
                .collect::<String>()
        }
    }).to_string()
}

/// Replace punctuation marks with standard TTS punctuation symbols
pub fn replace_punctuation(text: &str) -> String {
    let mut res = text.to_string();
    res = res.replace("嗯", "恩").replace("呣", "母");

    for (k, v) in REP_MAP.iter() {
        res = res.replace(k, v);
    }
    res
}

/// Collapse consecutive punctuation marks to a single punctuation
pub fn replace_consecutive_punctuation(text: &str) -> String {
    let replaced = replace_punctuation(text);
    let mut current = replaced;
    loop {
        let next = CONSECUTIVE_PUNCT.replace_all(&current, "$1").to_string();
        if next == current {
            break;
        }
        current = next;
    }
    current
}

/// Full text normalization pipeline for Chinese
pub fn normalize_chinese_text(text: &str) -> String {
    let text = normalize_numbers(text);
    let text = replace_punctuation(&text);
    let text = replace_consecutive_punctuation(&text);
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_to_chinese() {
        assert_eq!(number_to_chinese(0), "零");
        assert_eq!(number_to_chinese(5), "五");
        assert_eq!(number_to_chinese(12), "十二");
        assert_eq!(number_to_chinese(105), "一百零五");
        assert_eq!(number_to_chinese(2024), "二千零二十四");
    }

    #[test]
    fn test_replace_consecutive_punctuation() {
        assert_eq!(replace_consecutive_punctuation("你好！！！"), "你好!");
        assert_eq!(replace_consecutive_punctuation("真的嗎？？？..."), "真的嗎?");
    }
}
