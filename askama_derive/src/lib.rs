#![deny(elided_lifetimes_in_paths)]
#![deny(unreachable_pub)]

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use proc_macro::TokenStream;
use proc_macro2::Span;

use parser::{generate_error_info, strip_common, ErrorInfo, ParseError};

mod config;
use config::{read_config_file, Config};
mod generator;
use generator::{Generator, MapChain};
mod heritage;
use heritage::{Context, Heritage};
mod input;
use input::{Print, TemplateArgs, TemplateInput};
#[cfg(test)]
mod tests;

#[proc_macro_derive(Template, attributes(template))]
pub fn derive_template(input: TokenStream) -> TokenStream {
    let ast = syn::parse::<syn::DeriveInput>(input).unwrap();
    match build_template(&ast) {
        Ok(source) => source.parse().unwrap(),
        Err(e) => {
            let mut e = e.into_compile_error();
            if let Ok(source) = build_skeleton(&ast) {
                let source: TokenStream = source.parse().unwrap();
                e.extend(source);
            }
            e
        }
    }
}

fn build_skeleton(ast: &syn::DeriveInput) -> Result<String, CompileError> {
    let template_args = TemplateArgs::fallback();
    let config = Config::new("", None, None)?;
    let input = TemplateInput::new(ast, &config, &template_args)?;
    let mut contexts = HashMap::new();
    let parsed = parser::Parsed::default();
    contexts.insert(&input.path, Context::empty(&parsed));
    Generator::new(&input, &contexts, None, MapChain::default()).build(&contexts[&input.path])
}

/// Takes a `syn::DeriveInput` and generates source code for it
///
/// Reads the metadata from the `template()` attribute to get the template
/// metadata, then fetches the source from the filesystem. The source is
/// parsed, and the parse tree is fed to the code generator. Will print
/// the parse tree and/or generated source according to the `print` key's
/// value as passed to the `template()` attribute.
pub(crate) fn build_template(ast: &syn::DeriveInput) -> Result<String, CompileError> {
    let template_args = TemplateArgs::new(ast)?;
    let config_path = template_args.config_path();
    let s = read_config_file(config_path)?;
    let config = Config::new(&s, config_path, template_args.whitespace.as_deref())?;
    let input = TemplateInput::new(ast, &config, &template_args)?;

    let mut templates = HashMap::new();
    input.find_used_templates(&mut templates)?;

    let mut contexts = HashMap::new();
    for (path, parsed) in &templates {
        contexts.insert(path, Context::new(input.config, path, parsed)?);
    }

    let ctx = &contexts[&input.path];
    let heritage = if !ctx.blocks.is_empty() || ctx.extends.is_some() {
        let heritage = Heritage::new(ctx, &contexts);

        if let Some(block_name) = input.block {
            if !heritage.blocks.contains_key(&block_name) {
                return Err(format!("cannot find block {}", block_name).into());
            }
        }

        Some(heritage)
    } else {
        None
    };

    if input.print == Print::Ast || input.print == Print::All {
        eprintln!("{:?}", templates[&input.path].nodes());
    }

    let code = Generator::new(&input, &contexts, heritage.as_ref(), MapChain::default())
        .build(&contexts[&input.path])?;
    if input.print == Print::Code || input.print == Print::All {
        eprintln!("{code}");
    }
    Ok(code)
}

#[derive(Debug, Clone)]
struct CompileError {
    msg: String,
    span: Span,
}

impl CompileError {
    fn new<S: fmt::Display>(msg: S, file_info: Option<FileInfo<'_, '_, '_>>) -> Self {
        let msg = match file_info {
            Some(file_info) => format!("{msg}{file_info}"),
            None => msg.to_string(),
        };
        Self {
            msg,
            span: Span::call_site(),
        }
    }

    fn into_compile_error(self) -> TokenStream {
        syn::Error::new(self.span, self.msg)
            .to_compile_error()
            .into()
    }
}

impl std::error::Error for CompileError {}

impl fmt::Display for CompileError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(&self.msg)
    }
}

impl From<ParseError> for CompileError {
    #[inline]
    fn from(e: ParseError) -> Self {
        Self {
            msg: e.to_string(),
            span: Span::call_site(),
        }
    }
}

impl From<&'static str> for CompileError {
    #[inline]
    fn from(s: &'static str) -> Self {
        Self {
            msg: s.into(),
            span: Span::call_site(),
        }
    }
}

impl From<String> for CompileError {
    #[inline]
    fn from(s: String) -> Self {
        Self {
            msg: s,
            span: Span::call_site(),
        }
    }
}

struct FileInfo<'a, 'b, 'c> {
    path: &'a Path,
    source: Option<&'b str>,
    node_source: Option<&'c str>,
}

impl<'a, 'b, 'c> FileInfo<'a, 'b, 'c> {
    fn new(path: &'a Path, source: Option<&'b str>, node_source: Option<&'c str>) -> Self {
        Self {
            path,
            source,
            node_source,
        }
    }
}

impl<'a, 'b, 'c> fmt::Display for FileInfo<'a, 'b, 'c> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.source, self.node_source) {
            (Some(source), Some(node_source)) => {
                let (
                    ErrorInfo {
                        row,
                        column,
                        source_after,
                    },
                    file_path,
                ) = generate_error_info(source, node_source, self.path);
                write!(
                    f,
                    "\n  --> {file_path}:{row}:{column}\n{source_after}",
                    row = row + 1
                )
            }
            _ => {
                let file_path = match std::env::current_dir() {
                    Ok(cwd) => strip_common(&cwd, self.path),
                    Err(_) => self.path.display().to_string(),
                };
                write!(f, "\n --> {file_path}")
            }
        }
    }
}

// This is used by the code generator to decide whether a named filter is part of
// Askama or should refer to a local `filters` module. It should contain all the
// filters shipped with Askama, even the optional ones (since optional inclusion
// in the const vector based on features seems impossible right now).
const BUILT_IN_FILTERS: &[&str] = &[
    "abs",
    "capitalize",
    "center",
    "e",
    "escape",
    "filesizeformat",
    "fmt",
    "format",
    "indent",
    "into_f64",
    "into_isize",
    "join",
    "linebreaks",
    "linebreaksbr",
    "paragraphbreaks",
    "lower",
    "lowercase",
    "safe",
    "title",
    "trim",
    "truncate",
    "upper",
    "uppercase",
    "urlencode",
    "urlencode_strict",
    "wordcount",
    // optional features, reserve the names anyway:
    "json",
];

const CRATE: &str = if cfg!(feature = "with-actix-web") {
    "::askama_actix"
} else if cfg!(feature = "with-axum") {
    "::askama_axum"
} else if cfg!(feature = "with-rocket") {
    "::askama_rocket"
} else if cfg!(feature = "with-warp") {
    "::askama_warp"
} else {
    "::askama"
};
