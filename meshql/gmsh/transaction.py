import uuid
import gmsh
from typing import Optional, OrderedDict, Sequence

from pydantic import BaseModel
from meshql.gmsh.entity import Entity
from meshql.mesh.loaders import load_from_gmsh
from meshql.utils.types import OrderedSet
import meshly

class GmshTransaction(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }

    is_commited: bool = False
    is_generated: bool = False
    id: uuid.UUID = uuid.uuid4()

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

    



class SingleEntityTransaction(GmshTransaction):
    entity: Entity
    "The entity that transaction will be applied towards"



class MultiEntityTransaction(GmshTransaction):
    entities: OrderedSet[Entity]
    "The entities that transaction will be applied towards"

    ref_id: Optional[str] = None
    "An optional reference ID for grouping related transactions"


class GmshTransactionContext:
    def __init__(self):
        self.entity_transactions = OrderedDict[
            tuple[type[GmshTransaction], Entity], GmshTransaction
        ]()
        self.system_transactions = OrderedDict[type[GmshTransaction], GmshTransaction](
        )

        self.mesh: Optional[meshly.Mesh] = None

    def get_transaction(
        self, transaction_type: type[GmshTransaction], entity: Optional[Entity] = None
    ) -> Optional[GmshTransaction]:
        if entity is None:
            transaction = self.system_transactions.get(transaction_type)
        else:
            transaction = self.entity_transactions.get((transaction_type.__name__, entity))

        assert isinstance(transaction, transaction_type), f"Transaction must be {transaction_type.__name__} instead was {type(transaction).__name__}"
        return transaction


    def add_transaction(
        self, transaction: GmshTransaction, ignore_duplicates: bool = False
    ):
        """
        transaction: Transaction - transaction to be added
        ignore_duplicates: bool = False - if True, will ignore duplicate entity transactions
        """
        if isinstance(transaction, (SingleEntityTransaction, MultiEntityTransaction)):
            entities = (
                transaction.entities
                if isinstance(transaction, MultiEntityTransaction)
                else OrderedSet([transaction.entity])
            )
            for entity in entities:
                is_multi_with_ref = isinstance(transaction, MultiEntityTransaction) and transaction.ref_id
                if is_multi_with_ref:
                    transaction_id = (transaction.__class__.__name__, str(transaction.ref_id))                    
                else:
                    transaction_id = (transaction.__class__.__name__, entity)

                if transaction_id not in self.entity_transactions:
                    self.entity_transactions[transaction_id] = transaction
                else:
                    old_transaction = self.entity_transactions[transaction_id]
 
                    if is_multi_with_ref:
                        old_transaction.entities.add(entity)
                    elif not ignore_duplicates:
                        if isinstance(old_transaction, MultiEntityTransaction):
                            old_transaction.entities.remove(entity)
                        self.entity_transactions[transaction_id] = transaction



        else:
            self.system_transactions[type(transaction)] = transaction

    def add_transactions(
        self, transactions: Sequence[GmshTransaction], ignore_duplicates: bool = False
    ):
        """
        transactions: Sequence[Transaction] - transactions to be added
        ignore_duplicates: bool = False - if True, will ignore duplicate entity transactions
        """
        for transaction in transactions:
            self.add_transaction(transaction, ignore_duplicates)

    def generate(self, dim: int = 3):
        gmsh.model.occ.synchronize()
        transactions = OrderedSet(
            [*self.entity_transactions.values(), *self.system_transactions.values()]
        )
        for transaction in transactions:
            transaction.before_gen()

        gmsh.model.mesh.generate(dim)

        for transaction in transactions:
            transaction.after_gen()

        self.entity_transactions = OrderedDict()
        self.system_transactions = OrderedDict()
        self.mesh = load_from_gmsh()
