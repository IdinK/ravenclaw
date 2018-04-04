from datetime import datetime
from anytree import RenderTree, ContRoundStyle, ContStyle
from anytree import Node as BasicNode
import copy


class Node:
	def __init__(self, obj, name, description=None, parent=None, node=None, tree = None, replace_existing = False):
		if tree is None:
			tree = {}
		self._tree = tree
		if name in self._tree:
			if not replace_existing:
				raise ValueError('name:', name, 'exists in the tree')
		self._name = name
		self._tree[name] = self
		if node is None:
			node = BasicNode(name=name, description=description)
		self._node = node
		if parent is not None:
			self.parent = parent
		else:
			self._parent = None
		self._obj = copy.deepcopy(obj)
		self._children = {}
		self._timestamp = datetime.now()
		self._description = description

	@property
	def obj(self):
		return copy.deepcopy(self._obj)

	@property
	def name(self):
		return self._name

	def add_child(self, child):
		if child.name==self.name:
			raise ValueError('child name:', child.name, 'is the same as parent name:', self.name)
		if child.name in self._children:
			raise ValueError('child name:', child.name, 'already exists among the children')
		self._children[child.name] = child

	def remove_child(self, child):
		if child.name not in self._children:
			raise ValueError('child name:', child.name, 'does not exist among the children')
		child._node.parent = None
		del self._children[child.name]

	@property
	def parent(self):
		return self._parent

	@parent.setter
	def parent(self, parent):
		self._parent = parent
		self.parent.add_child(self)
		self._node.parent = self.parent._node

	@property
	def children(self):
		return self._children

	@property
	def siblings(self):
		return {name:sibling for name,sibling in self.parent.children.items() if name!= self.name}

	def save_snapshot(self, obj, name, description = None):
		snapshot = Node(obj=obj, name=name, description=description, parent=self, tree=self._tree)
		return snapshot

	def revert(self):
		parent = self.parent
		parent.remove_child(child=self)
		del self._tree[self.name]
		return parent

	def get_snapshot(self, name):
		the_leaf = self._tree[name]
		return the_leaf

	def get_obj(self, snapshot_name = None):
		if snapshot_name is None:
			return self.obj
		else:
			return self.get_snapshot(name=snapshot_name).obj

	def tree_str(self, style = 'square', show_description=True):
		if style=='round':
			style = ContRoundStyle()
		else:
			style = ContStyle()
		result = ''
		for pre, fill, node in RenderTree(self._node.root, style=style):
			#print("%s%s" % (pre, node.name))
			if node.name is None and node != self._node:
				continue
			elif node.name is None:
				name = 'Current Node'
			else:
				name = node.name

			if result!='':
				result += '\n'

			result += pre + name # + '"' + fill + '"'
			if show_description and node.description is not None:
				result += ' ' + '"'+ node.description+ '"'


		return result

	def display(self, style='square', show_description=True):
		print(self.tree_str(style=style, show_description=show_description))


class Tree:
	def __init__(self, name, description=None, ):
		self._root = Node()