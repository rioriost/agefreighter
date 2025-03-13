#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import ply.lex as lex  # type: ignore
import ply.yacc as yacc  # type: ignore

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Reserved keywords for Cypher queries.
RESERVED = {
    "create": "CREATE",
    "delete": "DELETE",
    "detach": "DETACH",
    "exists": "EXISTS",
    "match": "MATCH",
    "merge": "MERGE",
    "optional": "OPTIONAL",
    "remove": "REMOVE",
    "return": "RETURN",
    "set": "SET",
    "union": "UNION",
    "unwind": "UNWIND",
    "with": "WITH",
    "limit": "LIMIT",
    "order": "ORDER",
    "skip": "SKIP",
    "where": "WHERE",
    "asc": "ASC",
    "ascending": "ASCENDING",
    "by": "BY",
    "desc": "DESC",
    "descending": "DESCENDING",
    "on": "ON",
    "all": "ALL",
    "case": "CASE",
    "else": "ELSE",
    "end": "END",
    "then": "THEN",
    "when": "WHEN",
    "and": "AND",
    "as": "AS",
    "contains": "CONTAINS",
    "distinct": "DISTINCT",
    "ends": "ENDS",
    "in": "IN",
    "is": "IS",
    "not": "NOT",
    "or": "OR",
    "starts": "STARTS",
    "xor": "XOR",
    "false": "FALSE",
    "null": "NULL",
    "true": "TRUE",
    "add": "ADD",
    "constraint": "CONSTRAINT",
    "do": "DO",
    "drop": "DROP",
    "for": "FOR",
    "mandatory": "MANDATORY",
    "of": "OF",
    "require": "REQUIRE",
    "scalar": "SCALAR",
    "unique": "UNIQUE",
    "call": "CALL",
    "yield": "YIELD",
    "cypher": "CYPHER",
}


class CypherLexer:
    """
    Lexer for the Cypher query language.
    """

    def __init__(self, log_level: int = logging.INFO):
        log.setLevel(log_level)

    # Build tokens list by combining basic tokens with reserved keywords.
    tokens = [
        "NUMBER",
        "STRING",
        "IDENTIFIER",
        "PARAM",
        "PERCENT",
        "STAR",
        "PLUS",
        "MINUS",
        "DOT",
        "SLASH",
        "LT",
        "LE",
        "EQ",
        "GT",
        "GE",
        "CARET",
        "ENDS_WITH",
        "IS_NOT_NULL",
        "IS_NULL",
        "STARTS_WITH",
        "LPAREN",
        "RPAREN",
        "LBRACKET",
        "RBRACKET",
        "LBRACE",
        "RBRACE",
        "COLON",
        "DASH",
        "COMMA",
        "ARROW",
        "PIPE",
        "PLUS_EQ",
    ] + list(RESERVED.values())

    # Simple token definitions.
    t_PERCENT = r"%"
    t_STAR = r"\*"
    t_PLUS = r"\+"
    t_DOT = r"\."
    t_SLASH = r"/"
    t_LT = r"<"
    t_LE = r"<="
    t_EQ = r"="
    t_GT = r">"
    t_GE = r">="
    t_CARET = r"\^"
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_LBRACKET = r"\["
    t_RBRACKET = r"\]"
    t_COMMA = r","
    t_LBRACE = r"\{"
    t_RBRACE = r"\}"
    t_COLON = r":"
    t_DASH = r"-"
    t_ARROW = r"->"
    t_PIPE = r"\|"
    t_PLUS_EQ = r"\+="

    # Complex token definitions.
    def t_ENDS_WITH(self, t):
        r"ENDS\s+WITH"
        t.type = "ENDS_WITH"
        return t

    def t_STARTS_WITH(self, t):
        r"STARTS\s+WITH"
        t.type = "STARTS_WITH"
        return t

    def t_IS_NOT_NULL(self, t):
        r"IS\s+NOT\s+NULL"
        t.type = "IS_NOT_NULL"
        return t

    def t_IS_NULL(self, t):
        r"IS\s+NULL"
        t.type = "IS_NULL"
        return t

    def t_NUMBER(self, t):
        r"\d+(\.\d+)?"
        t.value = float(t.value) if "." in t.value else int(t.value)
        return t

    def t_STRING(self, t):
        r"""('([^\\\n]|(\\.))*?')|("([^\\\n]|(\\.))*?")"""
        t.value = t.value[1:-1]  # Remove the surrounding quotes
        return t

    def t_BACKTICK_IDENTIFIER(self, t):
        r"`([^`\\]|(\\.))*`"
        t.value = t.value[1:-1]  # Remove the enclosing backticks.
        t.type = "IDENTIFIER"
        return t

    def t_IDENTIFIER(self, t):
        r"[A-Za-z_][A-Za-z_0-9]*"
        lower_val = t.value.lower()
        if lower_val in RESERVED:
            t.type = RESERVED[lower_val]
        return t

    def t_PARAM(self, t):
        r"\$[A-Za-z_][A-Za-z_0-9]*"
        # Optionally strip the '$' if you want just the identifier:
        t.value = t.value[1:]
        return t

    # Ignore spaces and tabs.
    t_ignore = " \t"

    def t_COMMENT(self, t):
        r"//.*"
        pass  # Skip comments

    def t_BLOCK_COMMENT(self, t):
        r"/\*(.|\n)*?\*/"
        t.lexer.lineno += t.value.count("\n")
        pass  # Skip block comments

    def t_newline(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)

    def build(self, **kwargs):
        """Build the lexer."""
        self.lexer = lex.lex(module=self, **kwargs)

    def input(self, data):
        """Input data for lexing."""
        self.lexer.input(data)

    def token(self):
        """Return the next token."""
        return self.lexer.token()


class CypherParser:
    """
    Parser for the Cypher query language.
    """

    tokens = CypherLexer.tokens

    # Define operator precedence.
    precedence = (
        ("right", "NOT"),
        ("left", "OR", "XOR"),
        ("left", "AND"),
        ("nonassoc", "EQ", "LT", "LE", "GT", "GE", "IN"),
        ("left", "PLUS", "MINUS"),
        ("left", "STAR", "SLASH", "PERCENT"),
        ("right", "CARET"),
        ("right", "UMINUS"),
    )

    def __init__(self, log_level: int = logging.INFO):
        log.setLevel(log_level)
        self.lexer = CypherLexer(log_level=log_level)
        self.lexer.build()
        self.parser = yacc.yacc(module=self)

    # Grammar Rules

    def p_query(self, p):
        """query : version_header query_part
        | query_part
        | query UNION query_part
        | query UNION ALL query_part"""
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 3:
            p[0] = p[2]  # Ignore version header if present.
        elif len(p) == 4:
            p[0] = ("UNION", p[1], p[3])
        else:  # UNION ALL case.
            p[0] = ("UNION_ALL", p[1], p[4])

    def p_version_header(self, p):
        "version_header : CYPHER NUMBER"
        # Optionally, you can store or ignore the version number.
        p[0] = ("version", p[2])

    def p_query_part(self, p):
        "query_part : clause_list"
        p[0] = p[1]

    def p_clause(self, p):
        """clause : match_clause
        | optional_match_clause
        | mandatory_match_clause
        | merge_clause
        | create_clause
        | with_clause
        | set_clause
        | delete_clause
        | detach_delete_clause
        | remove_clause
        | call_clause
        | return_clause
        | unwind_clause
        | order_clause"""
        p[0] = p[1]

    def p_clause_list(self, p):
        """clause_list : clause
        | clause_list clause"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_match_clause(self, p):
        """match_clause : MATCH pattern_spec where_clause_opt"""
        p[0] = ("MATCH", p[2], p[3])

    def p_optional_match_clause(self, p):
        """optional_match_clause : OPTIONAL MATCH pattern_spec where_clause_opt"""
        p[0] = ("OPTIONAL_MATCH", p[3], p[4])

    def p_create_clause(self, p):
        """create_clause : CREATE pattern_spec"""
        p[0] = ("CREATE", p[2])

    def p_return_clause(self, p):
        """return_clause : RETURN return_items"""
        p[0] = ("RETURN", p[2])

    def p_return_clause_distinct(self, p):
        """return_clause : RETURN DISTINCT return_items"""
        p[0] = ("RETURN_DISTINCT", p[3])

    def p_return_items(self, p):
        """return_items : return_item
        | return_items COMMA return_item"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_return_item(self, p):
        """return_item : expression"""
        p[0] = p[1]

    def p_return_item_alias(self, p):
        """return_item : expression AS IDENTIFIER"""
        p[0] = ("alias", p[1], p[3])

    def p_where_clause_opt(self, p):
        """where_clause_opt :
        | WHERE expression"""
        p[0] = p[2] if len(p) > 1 else None

    def p_pattern(self, p):
        """pattern : node_pattern pattern_chain_opt"""
        if p[2]:
            # Represent a chain as a starting node and a list of (relationship, node) tuples.
            p[0] = ("chain", p[1], p[2])
        else:
            p[0] = p[1]

    def p_pattern_chain_opt(self, p):
        """pattern_chain_opt :
        | pattern_chain_opt relationship_pattern node_pattern"""
        if len(p) == 1:
            p[0] = []
        else:
            p[0] = p[1] + [(p[2], p[3])]

    def p_pattern_list(self, p):
        """pattern_list : pattern
        | pattern_list COMMA pattern"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_node_pattern(self, p):
        "node_pattern : LPAREN node_content_opt RPAREN"
        p[0] = p[2]

    def p_node_content_opt(self, p):
        """node_content_opt :
        | node_content"""
        if len(p) == 1:
            # Empty node content: no identifier, no labels, no properties.
            p[0] = ("node", None, [], None)
        else:
            p[0] = p[1]

    def p_node_content(self, p):
        """node_content : IDENTIFIER labels_opt property_map_opt
        | property_map"""
        if len(p) == 2:
            # Anonymous node: only a property map.
            p[0] = ("node", None, [], p[1])
        else:
            p[0] = ("node", p[1], p[2], p[3])

    def p_node_content_param(self, p):
        "node_content : PARAM"
        p[0] = ("node_param", p[1])

    def p_node_content_with_param(self, p):
        "node_content : IDENTIFIER labels_opt PARAM"
        p[0] = ("node", p[1], p[2], ("param", p[3]))

    def p_node_content_anonymous(self, p):
        "node_content : COLON IDENTIFIER labels_opt property_map_opt"
        # p[2] is the first label, p[3] contains any additional labels.
        p[0] = ("node", None, [p[2]] + p[3], p[4])

    def p_property_access(self, p):
        "property_access : IDENTIFIER DOT IDENTIFIER"
        p[0] = ("property_access", p[1], p[3])

    def p_labels_opt(self, p):
        """labels_opt :
        | labels_opt COLON IDENTIFIER"""
        if len(p) == 1:
            p[0] = []
        else:
            p[0] = p[1] + [p[3]]

    def p_property_map_opt(self, p):
        """property_map_opt :
        | property_map"""
        p[0] = p[1] if len(p) > 1 else None

    def p_opt_relationship_type_list(self, p):
        """opt_relationship_type_list : relationship_type_list
        |"""
        if len(p) == 1:
            p[0] = []  # No relationship types specified.
        else:
            p[0] = p[1]

    def p_property_map(self, p):
        "property_map : LBRACE property_list RBRACE"
        p[0] = p[2]

    def p_property_list(self, p):
        """property_list : property
        | property_list COMMA property"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_pattern_spec(self, p):
        """pattern_spec : pattern_alias
        | pattern_list"""
        p[0] = p[1]

    def p_pattern_alias(self, p):
        """pattern_alias : IDENTIFIER EQ pattern"""
        p[0] = ("pattern_alias", p[1], p[3])

    def p_property(self, p):
        "property : IDENTIFIER COLON expression"
        p[0] = (p[1], p[3])

    def p_relationship_pattern(self, p):
        """relationship_pattern : DASH LBRACKET relationship_content RBRACKET ARROW
        | DASH LBRACKET relationship_content RBRACKET DASH"""
        if p[5] == "->":
            p[0] = ("directed", p[3])
        else:
            p[0] = ("undirected", p[3])

    def p_relationship_pattern_inbound(self, p):
        """relationship_pattern : LT DASH LBRACKET relationship_content RBRACKET DASH"""
        p[0] = ("directed_inbound", p[4])

    def p_relationship_content(self, p):
        "relationship_content : opt_relationship_type_list relationship_length_opt property_map_opt"
        p[0] = ("relationship", p[1], p[2], p[3])

    def p_relationship_type_list_single(self, p):
        "relationship_type_list : relationship_base"
        p[0] = [p[1]]

    def p_relationship_type_list_multiple(self, p):
        "relationship_type_list : relationship_type_list PIPE relationship_base"
        p[0] = p[1] + [p[3]]

    def p_relationship_base(self, p):
        """relationship_base : IDENTIFIER COLON reltype
        | COLON reltype
        | IDENTIFIER
        |"""
        if len(p) == 4:
            p[0] = {"variable": p[1], "type": p[3]}
        elif len(p) == 3:
            p[0] = {"variable": None, "type": p[2]}
        elif len(p) == 2:
            p[0] = {"variable": p[1], "type": None}
        else:
            p[0] = {"variable": None, "type": None}

    def p_relationship_length_opt(self, p):
        """relationship_length_opt : STAR
        | STAR relationship_length_spec
        |"""
        if len(p) == 1:
            p[0] = None
        elif len(p) == 2:
            p[0] = {"min": None, "max": None}  # No bounds specified.
        else:
            p[0] = p[2]

    def p_expression_binop_mod(self, p):
        "expression : expression PERCENT expression"
        p[0] = ("binop", "%", p[1], p[3])

    def p_expression_exponentiation(self, p):
        "expression : expression CARET expression"
        p[0] = ("binop", "^", p[1], p[3])

    def p_relationship_length_spec(self, p):
        """relationship_length_spec : NUMBER
        | NUMBER DOT DOT NUMBER
        | NUMBER DOT DOT"""
        if len(p) == 2:
            p[0] = {"min": int(p[1]), "max": int(p[1])}
        elif len(p) == 5:
            p[0] = {"min": int(p[1]), "max": int(p[4])}
        elif len(p) == 4:
            p[0] = {"min": int(p[1]), "max": None}

    def p_expression_binop(self, p):
        """expression : expression EQ expression
        | expression PLUS expression
        | expression DASH expression
        | expression STAR expression
        | expression SLASH expression
        | expression LT expression
        | expression LE expression
        | expression GT expression
        | expression GE expression"""
        p[0] = ("binop", p[2], p[1], p[3])

    def p_expression_list(self, p):
        "expression : LBRACKET list_items_opt RBRACKET"
        p[0] = ("list", p[2])

    def p_list_items_opt(self, p):
        """list_items_opt :
        | list_items"""
        p[0] = p[1] if len(p) > 1 else []

    def p_list_items(self, p):
        """list_items : expression
        | list_items COMMA expression"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_expression_logical(self, p):
        """expression : expression AND expression
        | expression OR expression
        | expression XOR expression"""
        p[0] = ("logical", p[2], p[1], p[3])

    def p_expression_contains(self, p):
        """expression : expression CONTAINS expression"""
        p[0] = ("contains", p[1], p[3])

    def p_expression_ends_starts(self, p):
        """expression : expression ENDS_WITH expression
        | expression STARTS_WITH expression"""
        if p[2] == "ENDS WITH":
            p[0] = ("ends_with", p[1], p[3])
        else:
            p[0] = ("starts_with", p[1], p[3])

    def p_expression_isnull(self, p):
        """expression : expression IS_NOT_NULL
        | expression IS_NULL"""
        if p[2] == "IS NOT NULL":
            p[0] = ("is_not_null", p[1])
        else:
            p[0] = ("is_null", p[1])

    def p_expression_property(self, p):
        "expression : expression DOT IDENTIFIER"
        p[0] = ("property", p[1], p[3])

    def p_expression_uminus(self, p):
        "expression : DASH expression %prec UMINUS"
        p[0] = ("uminus", p[2])

    def p_expression_group(self, p):
        "expression : LPAREN expression RPAREN"
        p[0] = p[2]

    def p_expression_number(self, p):
        "expression : NUMBER"
        p[0] = p[1]

    def p_expression_string(self, p):
        "expression : STRING"
        p[0] = p[1]

    def p_expression_identifier(self, p):
        "expression : IDENTIFIER"
        p[0] = p[1]

    def p_expression_index(self, p):
        "expression : expression LBRACKET expression RBRACKET"
        p[0] = ("index", p[1], p[3])

    def p_expression_function(self, p):
        "expression : function_call"
        p[0] = p[1]

    def p_expression_null(self, p):
        "expression : NULL"
        p[0] = None

    def p_expression_param(self, p):
        "expression : PARAM"
        p[0] = ("param", p[1])

    def p_function_call(self, p):
        "function_call : IDENTIFIER LPAREN arg_list RPAREN"
        p[0] = ("func_call", p[1], p[3])

    def p_arg_list(self, p):
        """arg_list :
        | arguments"""
        p[0] = p[1] if len(p) > 1 else []

    def p_arguments(self, p):
        """arguments : expression
        | arguments COMMA expression"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_expression_error(self, p):
        "expression : error"
        raise ValueError("Syntax error in expression!")
        p[0] = None

    def p_error(self, p):
        if p:
            raise ValueError(f"Syntax error at '{p.value}'")
        else:
            raise ValueError("Syntax error at EOF")

    def p_with_clause(self, p):
        """with_clause : WITH with_items
        | WITH with_items WHERE expression"""
        if len(p) == 3:
            p[0] = ("WITH", p[2])
        else:
            p[0] = ("WITH", p[2], p[4])

    def p_with_items(self, p):
        """with_items : with_item
        | with_items COMMA with_item"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_with_item(self, p):
        """with_item : expression
        | expression AS IDENTIFIER"""
        p[0] = p[1] if len(p) == 2 else ("alias", p[1], p[3])

    def p_set_clause(self, p):
        "set_clause : SET set_items"
        p[0] = ("SET", p[2])

    def p_set_items(self, p):
        """set_items : set_item
        | set_items COMMA set_item"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_set_item(self, p):
        "set_item : property_access EQ expression"
        p[0] = ("set", p[1], p[3])

    def p_set_item_node(self, p):
        "set_item : IDENTIFIER EQ expression"
        p[0] = ("set_node", p[1], p[3])

    def p_slice_spec(self, p):
        "slice_spec : expression DOT DOT expression"
        p[0] = (p[1], p[4])

    def p_expression_slice(self, p):
        "expression : expression LBRACKET slice_spec RBRACKET"
        p[0] = ("slice", p[1], p[3][0], p[3][1])

    def p_expression_star(self, p):
        "expression : STAR"
        p[0] = ("star",)

    def p_slice_spec_optional(self, p):
        "slice_spec : expression DOT DOT"
        p[0] = (p[1], None)

    def p_slice_spec_lower_omitted(self, p):
        "slice_spec : DOT DOT expression"
        p[0] = (None, p[3])

    def p_expression_list_comprehension(self, p):
        """expression : LBRACKET IDENTIFIER IN expression list_comprehension_tail_opt RBRACKET"""
        p[0] = ("list_comprehension", p[2], p[4], p[5][0], p[5][1])

    def p_list_comprehension_tail_opt(self, p):
        """list_comprehension_tail_opt : where_clause_opt
        | where_clause_opt PIPE expression"""
        if len(p) == 2:
            p[0] = (p[1], None)
        else:
            p[0] = (p[1], p[3])

    def p_list_comprehension_tail(self, p):
        "list_comprehension_tail : where_clause_opt PIPE expression"
        p[0] = (p[1], p[3])

    def p_expression_pattern(self, p):
        "expression : pattern"
        p[0] = ("pattern_expr", p[1])

    def p_expression_pattern_comprehension(self, p):
        "expression : LBRACKET pattern where_clause_opt PIPE expression RBRACKET"
        p[0] = ("pattern_comprehension", p[2], p[3], p[5])

    def p_expression_label_check(self, p):
        "expression : IDENTIFIER COLON IDENTIFIER"
        p[0] = ("label_check", p[1], p[3])

    def p_expression_map(self, p):
        "expression : LBRACE property_list RBRACE"
        p[0] = ("map", p[2])

    def p_expression_empty_map(self, p):
        "expression : LBRACE RBRACE"
        p[0] = ("map", [])

    def p_expression_map_projection(self, p):
        "expression : IDENTIFIER LBRACE projection_items RBRACE"
        p[0] = ("map_projection", p[1], p[3])

    def p_projection_items(self, p):
        """projection_items : projection_item
        | projection_items COMMA projection_item"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_projection_item_shorthand(self, p):
        "projection_item : DOT IDENTIFIER"
        p[0] = ("projection_shorthand", p[2])

    def p_projection_item_alias(self, p):
        "projection_item : IDENTIFIER COLON expression"
        p[0] = ("projection_alias", p[1], p[3])

    def p_projection_item_alias_expr(self, p):
        "projection_item : IDENTIFIER"
        p[0] = ("projection_alias", p[1], p[1])

    def p_projection_item_wildcard(self, p):
        "projection_item : DOT STAR"
        p[0] = ("projection_wildcard",)

    def p_expression_true(self, p):
        "expression : TRUE"
        p[0] = True

    def p_expression_false(self, p):
        "expression : FALSE"
        p[0] = False

    def p_unwind_clause(self, p):
        "unwind_clause : UNWIND expression AS IDENTIFIER"
        p[0] = ("UNWIND", p[2], p[4])

    def p_order_clause(self, p):
        "order_clause : ORDER BY order_items"
        p[0] = ("ORDER", p[3])

    def p_order_items(self, p):
        """order_items : order_item
        | order_items COMMA order_item"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_order_item(self, p):
        """order_item : expression
        | expression ASC
        | expression ASCENDING
        | expression DESC
        | expression DESCENDING"""
        if len(p) == 2:
            p[0] = ("order_item", p[1], "ASC")
        else:
            order = "ASC" if p[2].lower() in ("asc", "ascending") else "DESC"
            p[0] = ("order_item", p[1], order)

    def p_expression_case(self, p):
        """expression : CASE case_operand_opt case_when_clauses case_else_opt END"""
        p[0] = ("case", p[2], p[3], p[4])

    def p_expression_in(self, p):
        "expression : expression IN expression"
        p[0] = ("in", p[1], p[3])

    def p_case_operand_opt(self, p):
        """case_operand_opt :
        | expression"""
        p[0] = p[1] if len(p) > 1 else None

    def p_case_when_clause(self, p):
        """case_when_clause : WHEN expression THEN expression"""
        p[0] = (p[2], p[4])

    def p_case_when_clauses(self, p):
        """case_when_clauses : case_when_clause
        | case_when_clauses case_when_clause"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_case_else_opt(self, p):
        """case_else_opt :
        | ELSE expression"""
        p[0] = p[2] if len(p) > 1 else None

    def p_skip_clause(self, p):
        "clause : SKIP expression"
        p[0] = ("SKIP", p[2])

    def p_limit_clause(self, p):
        "clause : LIMIT expression"
        p[0] = ("LIMIT", p[2])

    def p_expression_distinct(self, p):
        "expression : DISTINCT expression"
        p[0] = ("distinct", p[2])

    def p_reltype(self, p):
        """reltype : IDENTIFIER
        | IN"""
        p[0] = p[1]

    def p_mandatory_match_clause(self, p):
        """mandatory_match_clause : MANDATORY MATCH pattern_spec where_clause_opt"""
        p[0] = ("MANDATORY_MATCH", p[3], p[4])

    def p_return_item_wildcard(self, p):
        "return_item : STAR"
        p[0] = ("wildcard",)

    def p_merge_clause(self, p):
        "merge_clause : MERGE pattern_spec"
        p[0] = ("MERGE", p[2])

    def p_expression_not(self, p):
        "expression : NOT expression"
        p[0] = ("not", p[2])

    def p_function_call_exists(self, p):
        "function_call : EXISTS LPAREN arg_list RPAREN"
        p[0] = ("func_call", "exists", p[3])

    def p_delete_clause(self, p):
        """delete_clause : DELETE delete_items
        | DETACH DELETE delete_items"""
        if len(p) == 3:
            p[0] = ("DELETE", p[2])
        else:
            p[0] = ("DETACH_DELETE", p[3])

    def p_delete_items(self, p):
        """delete_items : expression
        | delete_items COMMA expression"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_detach_delete_clause(self, p):
        "detach_delete_clause : DETACH DELETE delete_items"
        p[0] = ("DETACH_DELETE", p[3])

    def p_set_item_merge(self, p):
        "set_item : IDENTIFIER PLUS_EQ expression"
        p[0] = ("set_merge", p[1], p[3])

    def p_set_item_merge_property(self, p):
        "set_item : property_access PLUS_EQ expression"
        p[0] = ("set_merge_property", p[1], p[3])

    def p_set_item_label(self, p):
        "set_item : IDENTIFIER COLON IDENTIFIER"
        p[0] = ("set_label", p[1], p[3])

    def p_set_item_labels(self, p):
        "set_item : IDENTIFIER label_list"
        p[0] = ("set_labels", p[1], p[2])

    def p_label_list_single(self, p):
        "label_list : COLON IDENTIFIER"
        p[0] = [p[2]]

    def p_label_list_multiple(self, p):
        "label_list : label_list COLON IDENTIFIER"
        p[0] = p[1] + [p[3]]

    def p_remove_clause(self, p):
        "remove_clause : REMOVE remove_items"
        p[0] = ("REMOVE", p[2])

    def p_remove_items(self, p):
        """remove_items : remove_item
        | remove_items COMMA remove_item"""
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_remove_item_property(self, p):
        "remove_item : property_access"
        p[0] = ("remove_property", p[1])

    def p_remove_item_label(self, p):
        "remove_item : IDENTIFIER label_list"
        p[0] = ("remove_labels", p[1], p[2])

    def p_call_clause(self, p):
        """call_clause : CALL procedure_call yield_clause_opt call_where_clause_opt"""
        # You can structure the tuple to capture procedure call, yield (if any), and where (if any)
        p[0] = ("CALL", p[2], p[3], p[4])

    def p_procedure_call_with_args(self, p):
        "procedure_call : IDENTIFIER LPAREN arg_list RPAREN"
        p[0] = ("proc_call", p[1], p[3])

    def p_procedure_call_base(self, p):
        "procedure_call : IDENTIFIER"
        p[0] = p[1]

    def p_procedure_call_chain(self, p):
        "procedure_call : procedure_call DOT IDENTIFIER"
        p[0] = f"{p[1]}.{p[3]}"

    def p_procedure_call_args(self, p):
        "procedure_call : procedure_call LPAREN arg_list RPAREN"
        p[0] = ("proc_call", p[1], p[3])

    def p_yield_clause(self, p):
        "yield_clause : YIELD yield_items"
        p[0] = ("YIELD", p[2])

    def p_yield_items_single(self, p):
        "yield_items : yield_item"
        p[0] = [p[1]]

    def p_yield_items_multiple(self, p):
        "yield_items : yield_items COMMA yield_item"
        p[0] = p[1] + [p[3]]

    def p_yield_item(self, p):
        """yield_item : IDENTIFIER
        | IDENTIFIER AS IDENTIFIER"""
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ("alias", p[1], p[3])

    def p_yield_clause_opt(self, p):
        """yield_clause_opt :
        | yield_clause"""
        p[0] = p[1] if len(p) > 1 else None

    def p_call_where_clause_opt(self, p):
        """call_where_clause_opt :
        | WHERE expression"""
        p[0] = p[2] if len(p) > 1 else None

    def parse(self, data):
        """Parse the given Cypher query string."""
        return self.parser.parse(data, lexer=self.lexer.lexer)
