# Language
class tree_sitter.Language(ptr)
A class that defines how to parse a particular language.

Methods
copy()
Create a copy of the language.

field_id_for_name(name, /)
Get the numerical id for the given field name.

field_name_for_id(field_id, /)
Get the field name for the given numerical id.

id_for_node_kind(kind, named, /)
Get the numerical id for the given node kind.

lookahead_iterator(state, /)
Create a new LookaheadIterator for this language and parse state.

next_state(state, id, /)
Get the next parse state.

Tip

Combine this with lookahead_iterator to generate completion suggestions or valid symbols in error nodes.

Examples

state = language.next_state(node.parse_state, node.grammar_id)
node_kind_for_id(id, /)
Get the name of the node kind for the given numerical id.

node_kind_is_named(id, /)
Check if the node type for the given numerical id is named (as opposed to an anonymous node type).

node_kind_is_supertype(id, /)
Check if the node type for the given numerical id is a supertype.

Supertype nodes represent abstract categories of syntax nodes (e.g. “expression”).

node_kind_is_visible(id, /)
Check if the node type for the given numerical id is visible (as opposed to an auxiliary node type).

subtypes(supertype, /)
Get all subtype symbol IDs for a given supertype symbol.

Special Methods
__copy__()
Use copy.copy() to create a copy of the language.

__eq__(value, /)
Implements self==value.

__hash__()
Implements hash(self).

Important

On 32-bit platforms, you must use hash(self) & 0xFFFFFFFF to get the actual hash.

__ne__(value, /)
Implements self!=value.

__repr__()
Implements repr(self).

Attributes
abi_version
The ABI version number that indicates which version of the Tree-sitter CLI was used to generate this language.

field_count
The number of distinct field names in this language.

name
The name of the language.

node_kind_count
The number of distinct node types in this language.

parse_state_count
The number of valid states in this language.

semantic_version
The Semantic Version of the language.

supertypes
The supertype symbols of the language.

# LogType

class tree_sitter.LogType
Bases: IntEnum

The type of a log message.

Members
PARSE = 0
LEX = 1

# Lookaheaditerator
class tree_sitter.LookaheadIterator
Bases: Iterator

A class that is used to look up symbols valid in a specific parse state.

Tip

Lookahead iterators can be useful to generate suggestions and improve syntax error diagnostics.

To get symbols valid in an ERROR node, use the lookahead iterator on its first leaf node state. For MISSING nodes, a lookahead iterator created on the previous non-extra leaf node may be appropriate.

Methods
names()
Get a list of all symbol names.

reset(state, language=None)
Reset the lookahead iterator.

Returns
:
True if it was reset successfully or False if it failed.

symbols()
Get a list of all symbol IDs.

Special Methods
__iter__()
Implements iter(self).

__next__()
Implements next(self).

Attributes
current_symbol
The current symbol ID.

Newly created iterators will return the ERROR symbol.

current_symbol_name
The current symbol name.

language
The current language.

# Node
class tree_sitter.Node
A single node within a syntax Tree.

Methods
child(index, /)
Get this node’s child at the given index, where 0 represents the first child.

Caution

This method is fairly fast, but its cost is technically log(i), so if you might be iterating over a long list of children, you should use children or walk() instead.

child_by_field_id(id, /)
Get the first child with the given numerical field id.

Hint

You can convert a field name to an id using Language.field_id_for_name().

See also

child_by_field_name()

child_by_field_name(name, /)
Get the first child with the given field name.

child_with_descendant(descendant, /)
Get the node that contains the given descendant.

children_by_field_id(id, /)
Get a list of children with the given numerical field id.

See also

children_by_field_name()

children_by_field_name(name, /)
Get a list of children with the given field name.

descendant_for_byte_range(start_byte, end_byte, /)
Get the smallest node within this node that spans the given byte range.

descendant_for_point_range(start_point, end_point, /)
Get the smallest node within this node that spans the given point range.

edit(start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point)
Edit this node to keep it in-sync with source code that has been edited.

Note

This method is only rarely needed. When you edit a syntax tree via Tree.edit(), all of the nodes that you retrieve from the tree afterwards will already reflect the edit. You only need to use this when you have a specific Node instance that you want to keep and continue to use after an edit.

field_name_for_child(child_index, /)
Get the field name of this node’s child at the given index.

field_name_for_named_child()
field_name_for_child(self, child_index, /) –

Get the field name of this node’s named child at the given index.

first_child_for_byte(byte, /)
Get the node’s first child that contains or starts after the given byte offset.

first_named_child_for_byte(byte, /)
Get the node’s first named child that contains or starts after the given byte offset.

named_child(index, /)
Get this node’s named child at the given index, where 0 represents the first child.

Caution

This method is fairly fast, but its cost is technically log(i), so if you might be iterating over a long list of children, you should use children or walk() instead.

named_descendant_for_byte_range(start_byte, end_byte, /)
Get the smallest named node within this node that spans the given byte range.

named_descendant_for_point_range(start_point, end_point, /)
Get the smallest named node within this node that spans the given point range.

walk()
Create a new TreeCursor starting from this node.

Special Methods
__eq__(value, /)
Implements self==value.

__hash__()
Implements hash(self).

__ne__(value, /)
Implements self!=value.

__repr__()
Implements repr(self).

__str__()
Implements str(self).

Attributes
byte_range
The byte range of source code that this node represents, in terms of bytes.

child_count
This node’s number of children.

children
This node’s children.

Note

If you’re walking the tree recursively, you may want to use walk() instead.

descendant_count
This node’s number of descendants, including the node itself.

end_byte
The byte offset where this node ends.

end_point
This node’s end point.

grammar_id
This node’s type as a numerical id as it appears in the grammar ignoring aliases.

grammar_name
This node’s symbol name as it appears in the grammar ignoring aliases.

has_changes
Check if this node has been edited.

has_error
Check if this node represents a syntax error or contains any syntax errors anywhere within it.

id
This node’s numerical id.

Note

Within a given syntax tree, no two nodes have the same id. However, if a new tree is created based on an older tree, and a node from the old tree is reused in the process, then that node will have the same id in both trees.

is_error
Check if this node represents a syntax error.

Syntax errors represent parts of the code that could not be incorporated into a valid syntax tree.

is_extra
Check if this node is _extra_.

Extra nodes represent things which are not required by the grammar but can appear anywhere (e.g. whitespace).

is_missing
Check if this node is _missing_.

Missing nodes are inserted by the parser in order to recover from certain kinds of syntax errors.

is_named
Check if this node is _named_.

Named nodes correspond to named rules in the grammar, whereas anonymous nodes correspond to string literals in the grammar.

kind_id
This node’s type as a numerical id.

named_child_count
This node’s number of _named_ children.

named_children
This node’s _named_ children.

next_named_sibling
This node’s next named sibling.

next_parse_state
The parse state after this node.

next_sibling
This node’s next sibling.

parent
This node’s immediate parent.

parse_state
This node’s parse state.

prev_named_sibling
This node’s previous named sibling.

prev_sibling
This node’s previous sibling.

range
The range of source code that this node represents.

start_byte
The byte offset where this node starts.

start_point
This node’s start point

text
The text of the node, if the tree has not been edited

type
This node’s type as a string.

# Parser
class tree_sitter.Parser(language, *, included_ranges=None, timeout_micros=None)
A class that is used to produce a Tree based on some source code.

Methods
parse(source, /, old_tree=None, encoding='utf8')
Parse a slice of a bytestring or bytes provided in chunks by a callback.

The callback function takes a byte offset and position and returns a bytestring starting at that offset and position. The slices can be of any length. If the given position is at the end of the text, the callback should return an empty slice.

Returns
:
A Tree if parsing succeeded or None if the parser does not have an assigned language or the timeout expired.

print_dot_graphs(file)
Set the file descriptor to which the parser should write debugging graphs during parsing. The graphs are formatted in the DOT language. You can turn off this logging by passing None.

reset()
Instruct the parser to start the next parse from the beginning.

Note

If the parser previously failed because of a timeout, then by default, it will resume where it left off on the next call to parse(). If you don’t want to resume, and instead intend to use this parser to parse some other document, you must call reset() first.

Attributes
included_ranges
The ranges of text that the parser will include when parsing.

language
The language that will be used for parsing.

logger
The logger that the parser should use during parsing.

# Point
class tree_sitter.Point(row, column)
Bases: tuple

A position in a multi-line text document, in terms of rows and columns.

Methods
edit(start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point)
Edit this point to keep it in-sync with source code that has been edited.

Returns
:
The edited point and its new start byte.

Tip

This is useful for editing points without requiring a tree or node instance.

Added in version 0.26.0.

Special Methods
__repr__()
Implements repr(self).

Attributes
column
The zero-based column of the document.

Note

Measured in bytes.

row
The zero-based row of the document.

# Query
class tree_sitter.Query(language, source)
A set of patterns that match nodes in a syntax tree.

Raises
:
QueryError – If any error occurred while creating the query.

See also

Query Syntax

Note

The following predicates are supported by default:

#eq?, #not-eq?, #any-eq?, #any-not-eq?

#match?, #not-match?, #any-match?, #any-not-match?

#any-of?, #not-any-of?

#is?, #is-not?

#set!

Methods
capture_name(index)
Get the name of the capture at the given index.

capture_quantifier(pattern_index, capture_index)
Get the quantifier of the capture at the given indexes.

disable_capture(name)
Disable a certain capture within a query.

Important

Currently, there is no way to undo this.

disable_pattern(index)
Disable a certain pattern within a query.

Important

Currently, there is no way to undo this.

end_byte_for_pattern(index)
Get the byte offset where the given pattern ends in the query’s source.

is_pattern_guaranteed_at_step(index)
Check if a pattern is guaranteed to match once a given byte offset is reached.

is_pattern_non_local(index)
Check if the pattern with the given index is “non-local”.

Note

A non-local pattern has multiple root nodes and can match within a repeating sequence of nodes, as specified by the grammar. Non-local patterns disable certain optimizations that would otherwise be possible when executing a query on a specific range of a syntax tree.

is_pattern_rooted(index)
Check if the pattern with the given index has a single root node.

pattern_assertions(index)
Get the property assertions for the given pattern index.

Assertions are performed using the #is? and #is-not? predicates.

Returns
:
A dictionary of assertions, where the first item is the optional property value and the second item indicates whether the assertion was positive or negative.

pattern_settings(index)
Get the property settings for the given pattern index.

Properties are set using the #set! predicate.

Returns
:
A dictionary of properties with optional values.

start_byte_for_pattern(index)
Get the byte offset where the given pattern starts in the query’s source.

string_value(index)
Get the string literal at the given index.

Attributes
capture_count
The number of captures in the query.

pattern_count
The number of patterns in the query.

string_count
The number of string literals in the query.

# QueryCursor
class tree_sitter.QueryCursor(query, *, match_limit=None, timeout_micros=None)
A class for executing a Query on a syntax Tree.

Methods
captures(node, /, predicate=None, progress_callback=None)
Get a list of captures within the given node.

Returns
:
A dict where the keys are the names of the captures and the values are lists of the captured nodes.

Hint

This method returns all of the captures while matches() only returns the last match.

matches(node, /, predicate=None, progress_callback=None)
Get a list of matches within the given node.

Returns
:
A list of tuples where the first element is the pattern index and the second element is a dictionary that maps capture names to nodes.

set_byte_range(start, end)
Set the range of bytes in which the query will be executed.

Raises
:
ValueError – If the start byte exceeds the end byte.

Note

The query cursor will return matches that intersect with the given byte range. This means that a match may be returned even if some of its captures fall outside the specified range, as long as at least part of the match overlaps with it.

set_containing_byte_range(start, end)
Set the byte range within which all matches must be fully contained.

Raises
:
ValueError – If the start byte exceeds the end byte.

Note

In contrast to set_byte_range(), this will restrict the query cursor to only return matches where all nodes are fully contained within the given range. Both methods can be used together, e.g. to search for any matches that intersect line 5000, as long as they are fully contained within lines 4500-5500

Added in version 0.26.0.

set_containing_point_range(start, end)
Set the point range within which all matches must be fully contained.

Raises
:
ValueError – If the start point exceeds the end point.

Note

In contrast to set_point_range(), this will restrict the query cursor to only return matches where all nodes are fully contained within the given range. Both methods can be used together, e.g. to search for any matches that intersect line 5000, as long as they are fully contained within lines 4500-5500

Added in version 0.26.0.

set_max_start_depth(max_start_depth)
Set the maximum start depth for the query.

set_point_range(start, end)
Set the range of points in which the query will be executed.

Raises
:
ValueError – If the start point exceeds the end point.

Note

The query cursor will return matches that intersect with the given point range. This means that a match may be returned even if some of its captures fall outside the specified range, as long as at least part of the match overlaps with it.

Attributes
did_exceed_match_limit
Check if the query exceeded its maximum number of in-progress matches during its last execution.

match_limit
The maximum number of in-progress matches.

# QueryError

class tree_sitter.QueryError
Bases: ValueError

An error that occurred while attempting to create a Query.

# QueryPredicate
class tree_sitter.QueryPredicate
Bases: Protocol

A custom query predicate that runs on a pattern.

Special Methods
__call__(predicate, args, pattern_index, captures)
Parameters
:
predicate (str) – The name of the predicate.

args (list[tuple[str, Literal['capture', 'string']]]) – The arguments to the predicate.

pattern_index (int) – The index of the pattern within the query.

captures (dict[str, list[Node]]) – The captures contained in the pattern.

Returns
:
True if the predicate matches, False otherwise.

Tip

You don’t need to create an actual class, just a function with this signature.

# Range
class tree_sitter.Range(start_point, end_point, start_byte, end_byte)
A range of positions in a multi-line text document, both in terms of bytes and of rows and columns.

Methods
edit(start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point)
Edit this range to keep it in-sync with source code that has been edited.

Tip

This is useful for editing ranges without requiring a tree or node instance.

Added in version 0.26.0.

Special Methods
__eq__(value, /)
Implements self==value.

__ne__(value, /)
Implements self!=value.

__repr__()
Implements repr(self).

__hash__()
Implements hash(self).

Attributes
end_byte
The end byte.

end_point
The end point.

start_byte
The start byte.

start_point
The start point.

# Tree
class tree_sitter.Tree
A tree that represents the syntactic structure of a source code file.

Methods
changed_ranges(new_tree)
Compare this old edited syntax tree to a new syntax tree representing the same document, returning a sequence of ranges whose syntactic structure has changed.

Returns
:
Ranges where the hierarchical structure of syntax nodes (from root to leaf) has changed between the old and new trees. Characters outside these ranges have identical ancestor nodes in both trees.

Note

The returned ranges may be slightly larger than the exact changed areas, but Tree-sitter attempts to make them as small as possible.

Tip

For this to work correctly, this syntax tree must have been edited such that its ranges match up to the new tree.

Generally, you’ll want to call this method right after calling the Parser.parse() method. Call it on the old tree that was passed to the method, and pass the new tree that was returned from it.

copy()
Create a shallow copy of the tree.

edit(start_byte, old_end_byte, new_end_byte, start_point, old_end_point, new_end_point)
Edit the syntax tree to keep it in sync with source code that has been edited.

You must describe the edit both in terms of byte offsets and of row/column points.

print_dot_graph(file)
Write a DOT graph describing the syntax tree to the given file.

root_node_with_offset(offset_bytes, offset_extent, /)
Get the root node of the syntax tree, but with its position shifted forward by the given offset.

walk()
Create a new TreeCursor starting from the root of the tree.

Special Methods
__copy__()
Use copy.copy() to create a copy of the tree.

Attributes
included_ranges
The included ranges that were used to parse the syntax tree.

language
The language that was used to parse the syntax tree.

root_node
The root node of the syntax tree.

# TreeCursor
class tree_sitter.TreeCursor
A class for walking a syntax Tree efficiently.

Important

The cursor can only walk into children of the node it was constructed with.

Methods
copy()
Create an independent copy of the cursor.

goto_descendant(index, /)
Move the cursor to the node that is the n-th descendant of the original node that the cursor was constructed with, where 0 represents the original node itself.

goto_first_child()
Move this cursor to the first child of its current node.

Returns
:
True if the cursor successfully moved, or False if there were no children.

goto_first_child_for_byte(byte, /)
Move this cursor to the first child of its current node that contains or starts after the given byte offset.

Returns
:
The index of the child node if it was found, None otherwise.

goto_first_child_for_point(point, /)
Move this cursor to the first child of its current node that contains or starts after the given given row/column point.

Returns
:
The index of the child node if it was found, None otherwise.

goto_last_child()
Move this cursor to the last child of its current node.

Returns
:
True if the cursor successfully moved, or False if there were no children.

Caution

This method may be slower than goto_first_child() because it needs to iterate through all the children to compute the child’s position.

goto_next_sibling()
Move this cursor to the next sibling of its current node.

Returns
:
True if the cursor successfully moved, or False if there was no next sibling.

goto_parent()
Move this cursor to the parent of its current node.

Returns
:
True if the cursor successfully moved, or False if there was no parent node (i.e. the cursor was already on the root node).

goto_previous_sibling()
Move this cursor to the previous sibling of its current node.

Returns
:
True if the cursor successfully moved, or False if there was no previous sibling.

Caution

This method may be slower than goto_next_sibling() due to how node positions are stored. In the worst case, this will need to iterate through all the children up to the previous sibling node to recalculate its position.

reset(node, /)
Re-initialize the cursor to start at the original node that it was constructed with.

reset_to(cursor, /)
Re-initialize the cursor to the same position as another cursor.

Unlike reset(), this will not lose parent information and allows reusing already created cursors.

Special Methods
__copy__()
Use copy.copy() to create a copy of the cursor.

Attributes
depth
The depth of the cursor’s current node relative to the original node that it was constructed with.

descendant_index
The index of the cursor’s current node out of all of the descendants of the original node that the cursor was constructed with.

field_id
The numerical field id of this tree cursor’s current node, if available.

field_name
The field name of this tree cursor’s current node, if available.

node
The current node.
