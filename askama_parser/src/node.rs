use std::str;

use nom::branch::alt;
use nom::bytes::complete::{tag, take_till};
use nom::character::complete::char;
use nom::combinator::{
    complete, consumed, cut, eof, map, map_res, not, opt, peek, recognize, value,
};
use nom::error::ErrorKind;
use nom::error_position;
use nom::multi::{fold_many0, many0, many1, separated_list0, separated_list1};
use nom::sequence::{delimited, pair, preceded, terminated};
use nom::Parser;

use crate::{not_ws, ErrorContext, ParseResult};

use super::{
    bool_lit, char_lit, filter, identifier, is_ws, keyword, num_lit, path_or_identifier, skip_till,
    str_lit, ws, Expr, Filter, PathOrIdentifier, State,
};

#[derive(Debug, PartialEq)]
pub enum Node<'a> {
    Lit(Lit<'a>),
    Comment(Comment<'a>),
    Expr(Ws, Expr<'a>),
    Call(Call<'a>),
    Let(Let<'a>),
    If(If<'a>),
    Match(Match<'a>),
    Loop(Box<Loop<'a>>),
    Extends(Extends<'a>),
    BlockDef(BlockDef<'a>),
    Include(Include<'a>),
    Import(Import<'a>),
    Macro(Macro<'a>),
    Raw(Raw<'a>),
    Break(Ws),
    Continue(Ws),
    FilterBlock(FilterBlock<'a>),
}

impl<'a> Node<'a> {
    pub(super) fn many(i: &'a str, s: &State<'_>) -> ParseResult<'a, Vec<Self>> {
        complete(many0(alt((
            map(|i| Lit::parse(i, s), Self::Lit),
            map(|i| Comment::parse(i, s), Self::Comment),
            |i| Self::expr(i, s),
            |i| Self::parse(i, s),
        ))))
        .parse(i)
    }

    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        #[inline]
        fn wrap<'a, T>(
            func: impl FnOnce(T) -> Node<'a>,
            result: ParseResult<'a, T>,
        ) -> ParseResult<'a, Node<'a>> {
            result.map(|(i, n)| (i, func(n)))
        }

        let (j, tag) = preceded(
            |i| s.tag_block_start(i),
            peek(preceded(
                pair(opt(Whitespace::parse), take_till(not_ws)),
                identifier,
            )),
        )
        .parse(i)?;

        let func = match tag {
            "call" => |i, s| wrap(Self::Call, Call::parse(i, s)),
            "let" | "set" => |i, s| wrap(Self::Let, Let::parse(i, s)),
            "if" => |i, s| wrap(Self::If, If::parse(i, s)),
            "for" => |i, s| wrap(|n| Self::Loop(Box::new(n)), Loop::parse(i, s)),
            "match" => |i, s| wrap(Self::Match, Match::parse(i, s)),
            "extends" => |i, _s| wrap(Self::Extends, Extends::parse(i)),
            "include" => |i, _s| wrap(Self::Include, Include::parse(i)),
            "import" => |i, _s| wrap(Self::Import, Import::parse(i)),
            "block" => |i, s| wrap(Self::BlockDef, BlockDef::parse(i, s)),
            "macro" => |i, s| wrap(Self::Macro, Macro::parse(i, s)),
            "raw" => |i, s| wrap(Self::Raw, Raw::parse(i, s)),
            "break" => |i, s| Self::r#break(i, s),
            "continue" => |i, s| Self::r#continue(i, s),
            "filter" => |i, s| wrap(Self::FilterBlock, FilterBlock::parse(i, s)),
            _ => {
                return Err(ErrorContext::from_err(nom::Err::Error(error_position!(
                    i,
                    ErrorKind::Tag
                ))));
            }
        };

        let (i, _) = s.nest(j)?;
        let result = func(i, s);
        s.leave();
        let (i, node) = result?;

        let (i, closed) = cut(alt((
            value(true, |i| s.tag_block_end(i)),
            value(false, ws(eof)),
        )))
        .parse(i)?;
        match closed {
            true => Ok((i, node)),
            false => Err(ErrorContext::unclosed("block", s.syntax.block_end, i).into()),
        }
    }

    fn r#break(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("break")),
            opt(Whitespace::parse),
        );
        let (j, (pws, _, nws)) = p.parse(i)?;
        if !s.is_in_loop() {
            return Err(nom::Err::Failure(ErrorContext::new(
                "you can only `break` inside a `for` loop",
                i,
            )));
        }
        Ok((j, Self::Break(Ws(pws, nws))))
    }

    fn r#continue(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("continue")),
            opt(Whitespace::parse),
        );
        let (j, (pws, _, nws)) = p.parse(i)?;
        if !s.is_in_loop() {
            return Err(nom::Err::Failure(ErrorContext::new(
                "you can only `continue` inside a `for` loop",
                i,
            )));
        }
        Ok((j, Self::Continue(Ws(pws, nws))))
    }

    fn expr(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let (i, (pws, expr)) = preceded(
            |i| s.tag_expr_start(i),
            cut(pair(
                opt(Whitespace::parse),
                ws(|i| Expr::parse(i, s.level.get())),
            )),
        )
        .parse(i)?;

        let (i, (nws, closed)) = cut(pair(
            opt(Whitespace::parse),
            alt((value(true, |i| s.tag_expr_end(i)), value(false, ws(eof)))),
        ))
        .parse(i)?;
        match closed {
            true => Ok((i, Self::Expr(Ws(pws, nws), expr))),
            false => Err(ErrorContext::unclosed("expression", s.syntax.expr_end, i).into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target<'a> {
    Name(&'a str),
    Tuple(Vec<&'a str>, Vec<Target<'a>>),
    Struct(Vec<&'a str>, Vec<(&'a str, Target<'a>)>),
    NumLit(&'a str),
    StrLit(&'a str),
    CharLit(&'a str),
    BoolLit(&'a str),
    Path(Vec<&'a str>),
    OrChain(Vec<Target<'a>>),
}

impl<'a> Target<'a> {
    /// Parses multiple targets with `or` separating them
    pub(super) fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        map(
            separated_list1(ws(tag("or")), |i| {
                s.nest(i)?;
                let ret = Self::parse_one(i, s)?;
                s.leave();
                Ok(ret)
            }),
            |mut opts| match opts.len() {
                1 => opts.pop().unwrap(),
                _ => Self::OrChain(opts),
            },
        )
        .parse(i)
    }

    /// Parses a single target without an `or`, unless it is wrapped in parentheses.
    fn parse_one(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut opt_opening_paren = map(opt(ws(char('('))), |o| o.is_some());
        let mut opt_closing_paren = map(opt(ws(char(')'))), |o| o.is_some());
        let mut opt_opening_brace = map(opt(ws(char('{'))), |o| o.is_some());

        let (i, lit) = opt(Self::lit).parse(i)?;
        if let Some(lit) = lit {
            return Ok((i, lit));
        }

        // match tuples and unused parentheses
        let (i, target_is_tuple) = opt_opening_paren.parse(i)?;
        if target_is_tuple {
            let (i, is_empty_tuple) = opt_closing_paren.parse(i)?;
            if is_empty_tuple {
                return Ok((i, Self::Tuple(Vec::new(), Vec::new())));
            }

            let (i, first_target) = Self::parse(i, s)?;
            let (i, is_unused_paren) = opt_closing_paren.parse(i)?;
            if is_unused_paren {
                return Ok((i, first_target));
            }

            let mut targets = vec![first_target];
            let (i, _) = cut((
                fold_many0(
                    preceded(ws(char(',')), |i| Self::parse(i, s)),
                    || (),
                    |_, target| {
                        targets.push(target);
                    },
                ),
                opt(ws(char(','))),
                ws(cut(char(')'))),
            ))
            .parse(i)?;
            return Ok((i, Self::Tuple(Vec::new(), targets)));
        }

        let path = |i| {
            map_res(path_or_identifier, |v| match v {
                PathOrIdentifier::Path(v) => Ok(v),
                PathOrIdentifier::Identifier(v) => Err(v),
            })
            .parse(i)
        };

        // match structs
        let (i, path) = opt(path).parse(i)?;
        if let Some(path) = path {
            let i_before_matching_with = i;
            let (i, _) = opt(ws(keyword("with"))).parse(i)?;

            let (i, is_unnamed_struct) = opt_opening_paren.parse(i)?;
            if is_unnamed_struct {
                let (i, targets) = alt((
                    map(char(')'), |_| Vec::new()),
                    terminated(
                        cut(separated_list1(ws(char(',')), |i| Self::parse(i, s))),
                        pair(opt(ws(char(','))), ws(cut(char(')')))),
                    ),
                ))
                .parse(i)?;
                return Ok((i, Self::Tuple(path, targets)));
            }

            let (i, is_named_struct) = opt_opening_brace.parse(i)?;
            if is_named_struct {
                let (i, targets) = alt((
                    map(char('}'), |_| Vec::new()),
                    terminated(
                        cut(separated_list1(ws(char(',')), |i| Self::named(i, s))),
                        pair(opt(ws(char(','))), ws(cut(char('}')))),
                    ),
                ))
                .parse(i)?;
                return Ok((i, Self::Struct(path, targets)));
            }

            return Ok((i_before_matching_with, Self::Path(path)));
        }

        // neither literal nor struct nor path
        let (new_i, name) = identifier(i)?;
        Ok((new_i, Self::verify_name(i, name)?))
    }

    fn lit(i: &'a str) -> ParseResult<'a, Self> {
        alt((
            map(str_lit, Self::StrLit),
            map(char_lit, Self::CharLit),
            map(num_lit, Self::NumLit),
            map(bool_lit, Self::BoolLit),
        ))
        .parse(i)
    }

    fn named(init_i: &'a str, s: &State<'_>) -> ParseResult<'a, (&'a str, Self)> {
        let (i, (src, target)) = pair(
            identifier,
            opt(preceded(ws(char(':')), |i| Self::parse(i, s))),
        )
        .parse(init_i)?;

        let target = match target {
            Some(target) => target,
            None => Self::verify_name(init_i, src)?,
        };

        Ok((i, (src, target)))
    }

    fn verify_name(input: &'a str, name: &'a str) -> Result<Self, nom::Err<ErrorContext<'a>>> {
        match name {
            "self" | "writer" => Err(nom::Err::Failure(ErrorContext::new(
                format!("cannot use `{name}` as a name"),
                input,
            ))),
            _ => Ok(Self::Name(name)),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct When<'a> {
    pub ws: Ws,
    pub target: Target<'a>,
    pub nodes: Vec<Node<'a>>,
}

impl<'a> When<'a> {
    fn r#match(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            |i| s.tag_block_start(i),
            opt(Whitespace::parse),
            ws(keyword("else")),
            cut((
                opt(Whitespace::parse),
                |i| s.tag_block_end(i),
                cut(|i| Node::many(i, s)),
            )),
        );
        let (i, (_, pws, _, (nws, _, nodes))) = p.parse(i)?;
        Ok((
            i,
            Self {
                ws: Ws(pws, nws),
                target: Target::Name("_"),
                nodes,
            },
        ))
    }

    #[allow(clippy::self_named_constructors)]
    fn when(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            |i| s.tag_block_start(i),
            opt(Whitespace::parse),
            ws(keyword("when")),
            cut((
                ws(|i| Target::parse(i, s)),
                opt(Whitespace::parse),
                |i| s.tag_block_end(i),
                cut(|i| Node::many(i, s)),
            )),
        );
        let (i, (_, pws, _, (target, nws, _, nodes))) = p.parse(i)?;
        Ok((
            i,
            Self {
                ws: Ws(pws, nws),
                target,
                nodes,
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct Cond<'a> {
    pub ws: Ws,
    pub cond: Option<CondTest<'a>>,
    pub nodes: Vec<Node<'a>>,
}

impl<'a> Cond<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let (i, (_, pws, cond, nws, _, nodes)) = (
            |i| s.tag_block_start(i),
            opt(Whitespace::parse),
            alt((
                preceded(ws(keyword("else")), opt(|i| CondTest::parse(i, s))),
                preceded(
                    ws(keyword("elif")),
                    cut(map(|i| CondTest::parse_cond(i, s), Some)),
                ),
            )),
            opt(Whitespace::parse),
            cut(|i| s.tag_block_end(i)),
            cut(|i| Node::many(i, s)),
        )
            .parse(i)?;
        Ok((
            i,
            Self {
                ws: Ws(pws, nws),
                cond,
                nodes,
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct CondTest<'a> {
    pub target: Option<Target<'a>>,
    pub expr: Expr<'a>,
}

impl<'a> CondTest<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        preceded(ws(keyword("if")), cut(|i| Self::parse_cond(i, s))).parse(i)
    }

    fn parse_cond(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let (i, (target, expr)) = pair(
            opt(delimited(
                ws(alt((keyword("let"), keyword("set")))),
                ws(|i| Target::parse(i, s)),
                ws(char('=')),
            )),
            ws(|i| Expr::parse(i, s.level.get())),
        )
        .parse(i)?;
        Ok((i, Self { target, expr }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Whitespace {
    Preserve,
    Suppress,
    Minimize,
}

impl Whitespace {
    fn parse(i: &str) -> ParseResult<'_, Self> {
        alt((
            value(Self::Preserve, char('+')),
            value(Self::Suppress, char('-')),
            value(Self::Minimize, char('~')),
        ))
        .parse(i)
    }
}

#[derive(Debug, PartialEq)]
pub struct Loop<'a> {
    pub ws1: Ws,
    pub var: Target<'a>,
    pub iter: Expr<'a>,
    pub cond: Option<Expr<'a>>,
    pub body: Vec<Node<'a>>,
    pub ws2: Ws,
    pub else_nodes: Vec<Node<'a>>,
    pub ws3: Ws,
}

impl<'a> Loop<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        fn content<'a>(i: &'a str, s: &State<'_>) -> ParseResult<'a, Vec<Node<'a>>> {
            s.enter_loop();
            let result = Node::many(i, s);
            s.leave_loop();
            result
        }

        let if_cond = preceded(
            ws(keyword("if")),
            cut(ws(|i| Expr::parse(i, s.level.get()))),
        );

        let else_block = |i| {
            let mut p = preceded(
                ws(keyword("else")),
                cut((
                    opt(Whitespace::parse),
                    delimited(
                        |i| s.tag_block_end(i),
                        |i| Node::many(i, s),
                        |i| s.tag_block_start(i),
                    ),
                    opt(Whitespace::parse),
                )),
            );
            let (i, (pws, nodes, nws)) = p.parse(i)?;
            Ok((i, (pws, nodes, nws)))
        };

        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("for")),
            cut((
                ws(|i| Target::parse(i, s)),
                ws(keyword("in")),
                cut((
                    ws(|i| Expr::parse(i, s.level.get())),
                    opt(if_cond),
                    opt(Whitespace::parse),
                    |i| s.tag_block_end(i),
                    cut((
                        |i| content(i, s),
                        cut((
                            |i| s.tag_block_start(i),
                            opt(Whitespace::parse),
                            opt(else_block),
                            ws(keyword("endfor")),
                            opt(Whitespace::parse),
                        )),
                    )),
                )),
            )),
        );
        let (i, (pws1, _, (var, _, (iter, cond, nws1, _, (body, (_, pws2, else_block, _, nws2)))))) =
            p.parse(i)?;
        let (nws3, else_block, pws3) = else_block.unwrap_or_default();
        Ok((
            i,
            Self {
                ws1: Ws(pws1, nws1),
                var,
                iter,
                cond,
                body,
                ws2: Ws(pws2, nws3),
                else_nodes: else_block,
                ws3: Ws(pws3, nws2),
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct Macro<'a> {
    pub ws1: Ws,
    pub name: &'a str,
    pub args: Vec<&'a str>,
    pub nodes: Vec<Node<'a>>,
    pub ws2: Ws,
}

impl<'a> Macro<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        fn parameters(i: &str) -> ParseResult<'_, Vec<&str>> {
            delimited(
                ws(char('(')),
                separated_list0(char(','), ws(identifier)),
                (opt(ws(char(','))), char(')')),
            )
            .parse(i)
        }

        let mut start = (
            opt(Whitespace::parse),
            ws(keyword("macro")),
            cut((
                ws(identifier),
                opt(ws(parameters)),
                opt(Whitespace::parse),
                |i| s.tag_block_end(i),
            )),
        );
        let (j, (pws1, _, (name, params, nws1, _))) = start.parse(i)?;
        if name == "super" {
            return Err(nom::Err::Failure(ErrorContext::new(
                "'super' is not a valid name for a macro",
                i,
            )));
        }

        let mut end = cut((
            |i| Node::many(i, s),
            cut((
                |i| s.tag_block_start(i),
                opt(Whitespace::parse),
                ws(keyword("endmacro")),
                cut(preceded(
                    opt(|before| {
                        let (after, end_name) = ws(identifier).parse(before)?;
                        check_end_name(before, after, name, end_name, "macro")
                    }),
                    opt(Whitespace::parse),
                )),
            )),
        ));
        let (i, (contents, (_, pws2, _, nws2))) = end.parse(j)?;

        Ok((
            i,
            Self {
                ws1: Ws(pws1, nws1),
                name,
                args: params.unwrap_or_default(),
                nodes: contents,
                ws2: Ws(pws2, nws2),
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct FilterBlock<'a> {
    pub ws1: Ws,
    pub filters: Filter<'a>,
    pub nodes: Vec<Node<'a>>,
    pub ws2: Ws,
}

impl<'a> FilterBlock<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut start = (
            opt(Whitespace::parse),
            ws(keyword("filter")),
            cut((
                ws(identifier),
                opt(|i| Expr::arguments(i, s.level.get(), false)),
                many0(|i| filter(i, s.level.get())),
                ws(|i| Ok((i, ()))),
                opt(Whitespace::parse),
                |i| s.tag_block_end(i),
            )),
        );
        let (i, (pws1, _, (filter_name, params, extra_filters, _, nws1, _))) = start.parse(i)?;

        let mut filters = Filter {
            name: filter_name,
            arguments: params.unwrap_or_default(),
        };
        for (filter_name, args) in extra_filters {
            filters = Filter {
                name: filter_name,
                arguments: {
                    let mut args = args.unwrap_or_default();
                    args.insert(0, Expr::Filter(filters));
                    args
                },
            };
        }

        let mut end = cut((
            |i| Node::many(i, s),
            cut((
                |i| s.tag_block_start(i),
                opt(Whitespace::parse),
                ws(keyword("endfilter")),
                opt(Whitespace::parse),
            )),
        ));
        let (i, (nodes, (_, pws2, _, nws2))) = end.parse(i)?;

        Ok((
            i,
            Self {
                ws1: Ws(pws1, nws1),
                filters,
                nodes,
                ws2: Ws(pws2, nws2),
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct Import<'a> {
    pub ws: Ws,
    pub path: &'a str,
    pub scope: &'a str,
}

impl<'a> Import<'a> {
    fn parse(i: &'a str) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("import")),
            cut((
                ws(str_lit),
                ws(keyword("as")),
                cut(pair(ws(identifier), opt(Whitespace::parse))),
            )),
        );
        let (i, (pws, _, (path, _, (scope, nws)))) = p.parse(i)?;
        Ok((
            i,
            Self {
                ws: Ws(pws, nws),
                path,
                scope,
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct Call<'a> {
    pub ws: Ws,
    pub scope: Option<&'a str>,
    pub name: &'a str,
    pub args: Vec<Expr<'a>>,
}

impl<'a> Call<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("call")),
            cut((
                opt((ws(identifier), ws(tag("::")))),
                ws(identifier),
                opt(ws(|nested| Expr::arguments(nested, s.level.get(), true))),
                opt(Whitespace::parse),
            )),
        );
        let (i, (pws, _, (scope, name, args, nws))) = p.parse(i)?;
        let scope = scope.map(|(scope, _)| scope);
        let args = args.unwrap_or_default();
        Ok((
            i,
            Self {
                ws: Ws(pws, nws),
                scope,
                name,
                args,
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct Match<'a> {
    pub ws1: Ws,
    pub expr: Expr<'a>,
    pub arms: Vec<When<'a>>,
    pub ws2: Ws,
}

impl<'a> Match<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("match")),
            cut((
                ws(|i| Expr::parse(i, s.level.get())),
                opt(Whitespace::parse),
                |i| s.tag_block_end(i),
                cut((
                    ws(many0(ws(value((), |i| Comment::parse(i, s))))),
                    many1(|i| When::when(i, s)),
                    cut((
                        opt(|i| When::r#match(i, s)),
                        cut((
                            ws(|i| s.tag_block_start(i)),
                            opt(Whitespace::parse),
                            ws(keyword("endmatch")),
                            opt(Whitespace::parse),
                        )),
                    )),
                )),
            )),
        );
        let (i, (pws1, _, (expr, nws1, _, (_, arms, (else_arm, (_, pws2, _, nws2)))))) =
            p.parse(i)?;

        let mut arms = arms;
        if let Some(arm) = else_arm {
            arms.push(arm);
        }

        Ok((
            i,
            Self {
                ws1: Ws(pws1, nws1),
                expr,
                arms,
                ws2: Ws(pws2, nws2),
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct BlockDef<'a> {
    pub ws1: Ws,
    pub name: &'a str,
    pub nodes: Vec<Node<'a>>,
    pub ws2: Ws,
}

impl<'a> BlockDef<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut start = (
            opt(Whitespace::parse),
            ws(keyword("block")),
            cut((ws(identifier), opt(Whitespace::parse), |i| {
                s.tag_block_end(i)
            })),
        );
        let (i, (pws1, _, (name, nws1, _))) = start.parse(i)?;

        let mut end = cut((
            |i| Node::many(i, s),
            cut((
                |i| s.tag_block_start(i),
                opt(Whitespace::parse),
                ws(keyword("endblock")),
                cut((
                    opt(|before| {
                        let (after, end_name) = ws(identifier).parse(before)?;
                        check_end_name(before, after, name, end_name, "block")
                    }),
                    opt(Whitespace::parse),
                )),
            )),
        ));
        let (i, (nodes, (_, pws2, _, (_, nws2)))) = end.parse(i)?;

        Ok((
            i,
            BlockDef {
                ws1: Ws(pws1, nws1),
                name,
                nodes,
                ws2: Ws(pws2, nws2),
            },
        ))
    }
}

fn check_end_name<'a>(
    before: &'a str,
    after: &'a str,
    name: &'a str,
    end_name: &'a str,
    kind: &str,
) -> ParseResult<'a> {
    if name == end_name {
        return Ok((after, end_name));
    }

    Err(nom::Err::Failure(ErrorContext::new(
        match name.is_empty() && !end_name.is_empty() {
            true => format!("unexpected name `{end_name}` in `end{kind}` tag for unnamed `{kind}`"),
            false => format!("expected name `{name}` in `end{kind}` tag, found `{end_name}`"),
        },
        before,
    )))
}

#[derive(Debug, PartialEq)]
pub struct Lit<'a> {
    pub lws: &'a str,
    pub val: &'a str,
    pub rws: &'a str,
}

impl<'a> Lit<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let p_start = alt((
            tag(s.syntax.block_start),
            tag(s.syntax.comment_start),
            tag(s.syntax.expr_start),
        ));

        let (i, _) = not(eof).parse(i)?;
        let (i, content) = opt(recognize(skip_till(p_start))).parse(i)?;
        let (i, content) = match content {
            Some("") => {
                // {block,comment,expr}_start follows immediately.
                return Err(nom::Err::Error(error_position!(i, ErrorKind::TakeUntil)));
            }
            Some(content) => (i, content),
            None => ("", i), // there is no {block,comment,expr}_start: take everything
        };
        Ok((i, Self::split_ws_parts(content)))
    }

    pub(crate) fn split_ws_parts(s: &'a str) -> Self {
        let trimmed_start = s.trim_start_matches(is_ws);
        let len_start = s.len() - trimmed_start.len();
        let trimmed = trimmed_start.trim_end_matches(is_ws);
        Self {
            lws: &s[..len_start],
            val: trimmed,
            rws: &trimmed_start[trimmed.len()..],
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Raw<'a> {
    pub ws1: Ws,
    pub lit: Lit<'a>,
    pub ws2: Ws,
}

impl<'a> Raw<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let endraw = (
            |i| s.tag_block_start(i),
            opt(Whitespace::parse),
            ws(keyword("endraw")),
            opt(Whitespace::parse),
            peek(|i| s.tag_block_end(i)),
        );

        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("raw")),
            cut((
                opt(Whitespace::parse),
                |i| s.tag_block_end(i),
                consumed(skip_till(endraw)),
            )),
        );

        let (_, (pws1, _, (nws1, _, (contents, (i, (_, pws2, _, nws2, _)))))) = p.parse(i)?;
        let lit = Lit::split_ws_parts(contents);
        let ws1 = Ws(pws1, nws1);
        let ws2 = Ws(pws2, nws2);
        Ok((i, Self { ws1, lit, ws2 }))
    }
}

#[derive(Debug, PartialEq)]
pub struct Let<'a> {
    pub ws: Ws,
    pub var: Target<'a>,
    pub val: Option<Expr<'a>>,
}

impl<'a> Let<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            ws(alt((keyword("let"), keyword("set")))),
            cut((
                ws(|i| Target::parse(i, s)),
                opt(preceded(
                    ws(char('=')),
                    ws(|i| Expr::parse(i, s.level.get())),
                )),
                opt(Whitespace::parse),
            )),
        );
        let (i, (pws, _, (var, val, nws))) = p.parse(i)?;

        Ok((
            i,
            Let {
                ws: Ws(pws, nws),
                var,
                val,
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct If<'a> {
    pub ws: Ws,
    pub branches: Vec<Cond<'a>>,
}

impl<'a> If<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            |i| CondTest::parse(i, s),
            cut((
                opt(Whitespace::parse),
                |i| s.tag_block_end(i),
                cut((
                    |i| Node::many(i, s),
                    many0(|i| Cond::parse(i, s)),
                    cut((
                        |i| s.tag_block_start(i),
                        opt(Whitespace::parse),
                        ws(keyword("endif")),
                        opt(Whitespace::parse),
                    )),
                )),
            )),
        );

        let (i, (pws1, cond, (nws1, _, (nodes, elifs, (_, pws2, _, nws2))))) = p.parse(i)?;
        let mut branches = vec![Cond {
            ws: Ws(pws1, nws1),
            cond: Some(cond),
            nodes,
        }];
        branches.extend(elifs);

        Ok((
            i,
            Self {
                ws: Ws(pws2, nws2),
                branches,
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct Include<'a> {
    pub ws: Ws,
    pub path: &'a str,
}

impl<'a> Include<'a> {
    fn parse(i: &'a str) -> ParseResult<'a, Self> {
        let mut p = (
            opt(Whitespace::parse),
            ws(keyword("include")),
            cut(pair(ws(str_lit), opt(Whitespace::parse))),
        );
        let (i, (pws, _, (path, nws))) = p.parse(i)?;
        Ok((
            i,
            Self {
                ws: Ws(pws, nws),
                path,
            },
        ))
    }
}

#[derive(Debug, PartialEq)]
pub struct Extends<'a> {
    pub path: &'a str,
}

impl<'a> Extends<'a> {
    fn parse(i: &'a str) -> ParseResult<'a, Self> {
        let start = i;

        let (i, (pws, _, (path, nws))) = (
            opt(Whitespace::parse),
            ws(keyword("extends")),
            cut(pair(ws(str_lit), opt(Whitespace::parse))),
        )
            .parse(i)?;
        match (pws, nws) {
            (None, None) => Ok((i, Self { path })),
            (_, _) => Err(nom::Err::Failure(ErrorContext::new(
                "whitespace control is not allowed on `extends`",
                start,
            ))),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Comment<'a> {
    pub ws: Ws,
    pub content: &'a str,
}

impl<'a> Comment<'a> {
    fn parse(i: &'a str, s: &State<'_>) -> ParseResult<'a, Self> {
        #[derive(Debug, Clone, Copy)]
        enum Tag {
            Open,
            Close,
        }

        fn tag<'a>(i: &'a str, s: &State<'_>) -> ParseResult<'a, Tag> {
            alt((
                value(Tag::Open, |i| s.tag_comment_start(i)),
                value(Tag::Close, |i| s.tag_comment_end(i)),
            ))
            .parse(i)
        }

        fn content<'a>(mut i: &'a str, s: &State<'_>) -> ParseResult<'a, ()> {
            let mut depth = 0usize;
            loop {
                let (_, tag) = opt(skip_till(|i| tag(i, s))).parse(i)?;
                let Some((j, tag)) = tag else {
                    return Err(ErrorContext::unclosed("comment", s.syntax.comment_end, i).into());
                };
                match tag {
                    Tag::Open => match depth.checked_add(1) {
                        Some(new_depth) => depth = new_depth,
                        None => {
                            return Err(nom::Err::Failure(ErrorContext::new(
                                "too deeply nested comments",
                                i,
                            )));
                        }
                    },
                    Tag::Close => match depth.checked_sub(1) {
                        Some(new_depth) => depth = new_depth,
                        None => return Ok((j, ())),
                    },
                }
                i = j;
            }
        }

        let (i, (pws, content)) = pair(
            preceded(|i| s.tag_comment_start(i), opt(Whitespace::parse)),
            recognize(cut(|i| content(i, s))),
        )
        .parse(i)?;

        let mut nws = None;
        if let Some(content) = content.strip_suffix(s.syntax.comment_end) {
            nws = match content.chars().last() {
                Some('-') => Some(Whitespace::Suppress),
                Some('+') => Some(Whitespace::Preserve),
                Some('~') => Some(Whitespace::Minimize),
                _ => None,
            }
        };

        Ok((
            i,
            Self {
                ws: Ws(pws, nws),
                content,
            },
        ))
    }
}

/// First field is "minus/plus sign was used on the left part of the item".
///
/// Second field is "minus/plus sign was used on the right part of the item".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ws(pub Option<Whitespace>, pub Option<Whitespace>);
