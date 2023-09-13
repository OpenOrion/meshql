import uuid
import gmsh
from dataclasses import dataclass
from typing import Optional, OrderedDict, Sequence, Union
from meshql.entity import Entity
from meshql.mesh.importers import import_from_gmsh
from meshql.mesh.mesh import Mesh
from meshql.utils.types import OrderedSet

@dataclass
class Transaction:
    def __post_init__(self):
        self.is_commited: bool = False
        self.is_generated: bool = False
        self.id = uuid.uuid4()
    
    def before_gen(self):
        "completes transaction before mesh generation."
        ...

    def after_gen(self):
        "completes transaction after mesh generation."
        ...

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __eq__(self, __value: object) -> bool:
        return self.id.__eq__(__value)
    


@dataclass(eq=False)
class SingleEntityTransaction(Transaction):
    entity: Entity
    "The entity that transaction will be applied towards"

@dataclass(eq=False)
class MultiEntityTransaction(Transaction):
    entities: OrderedSet[Entity]
    "The entities that transaction will be applied towards"


class TransactionContext:
    def __init__(self):
        self.entity_transactions = OrderedDict[tuple[type[Transaction], Entity], Transaction]()
        self.system_transactions = OrderedDict[type[Transaction], Transaction]()

        self.mesh: Optional[Mesh] = None
    
    def get_transaction(self, transaction_type: type[Transaction], entity: Optional[Entity] = None) -> Optional[Transaction]:
        if entity is None:
            return self.system_transactions.get(transaction_type)
        else:
            return self.entity_transactions.get((transaction_type, entity))


    def add_transaction(self, transaction: Transaction, ignore_duplicates: bool = False):
        """
        transaction: Transaction - transaction to be added
        ignore_duplicates: bool = False - if True, will ignore duplicate entity transactions
        """
        if isinstance(transaction, (SingleEntityTransaction, MultiEntityTransaction) ):
            entities = transaction.entities if isinstance(transaction, MultiEntityTransaction) else OrderedSet([transaction.entity])
            for entity in entities:
                transaction_id = (type(transaction), entity)
                if transaction_id not in self.entity_transactions:
                    self.entity_transactions[transaction_id] = transaction
                else:
                    if not ignore_duplicates:
                        old_transaction = self.entity_transactions[transaction_id]
                        if isinstance(old_transaction, MultiEntityTransaction):
                            old_transaction.entities.remove(entity)
                        self.entity_transactions[transaction_id] = transaction
        else:
            self.system_transactions[type(transaction)] = transaction

    def add_transactions(self, transactions: Sequence[Transaction], ignore_duplicates: bool = False):
        """
        transactions: Sequence[Transaction] - transactions to be added
        ignore_duplicates: bool = False - if True, will ignore duplicate entity transactions
        """
        for transaction in transactions:
            self.add_transaction(transaction, ignore_duplicates)

    def generate(self, dim: int = 3):
        gmsh.model.occ.synchronize()
        transactions = OrderedSet([
            *self.entity_transactions.values(),
            * self.system_transactions.values()
        ])
        for transaction in transactions:
            transaction.before_gen()

        gmsh.model.mesh.generate(dim)

        for transaction in transactions:
            transaction.after_gen()
        
        self.entity_transactions = OrderedDict()
        self.system_transactions = OrderedDict()
        self.mesh = import_from_gmsh()
