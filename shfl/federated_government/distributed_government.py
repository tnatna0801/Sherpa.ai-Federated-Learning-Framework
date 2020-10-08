import abc


class DistributedGovernment(abc.ABC):
    """
    Abstract class for distributed government

    # Arguments:
       model_builder_server: Function that returns the server's trainable model (see: [Model](../model))
       model_builder_node: Function that returns the node's trainable model
       federated_data: Federated data to use.
       (see: [FederatedData](../private/federated_operation/#federateddata-class))
       aggregator: Federated aggregator function (see: [Federated Aggregator](../federated_aggregator))
       model_param_access: Policy to access model's parameters, by default non-protected
       (see: [DataAccessDefinition](../private/data/#dataaccessdefinition-class))

    """

    def __init__(self, model_builder_server, model_builder_node, federated_data, aggregator, model_params_access=None):
        self._federated_data = federated_data
        self._model = model_builder_server()
        self._aggregator = aggregator
        for data_node in federated_data:
            data_node.model = model_builder_node()
            if model_params_access is not None:
                data_node.configure_model_params_access(model_params_access)

    @property
    def global_model(self):
        return self._model

    @abc.abstractmethod
    def evaluate_global_model(self, data_test, label_test):
        """
        Abstract method that evaluates the performance of the global model.

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """

    @abc.abstractmethod
    def deploy_central_model(self):
        """
        Abstract method to deploy the global learning model to each client (node) in the simulation.
        """

    @abc.abstractmethod
    def evaluate_clients(self, data_test, label_test):
        """
        Method that must implement every federated government extending this class.

        It evaluates the local learning models over global test dataset.

        # Arguments:
            test_data: test dataset
            test_label: corresponding labels to test dataset
        """

    def train_all_clients(self):
        """
        Train all the clients
        """
        for data_node in self._federated_data:
            data_node.train_model()

    def aggregate_weights(self):
        """
        Collect the weights from all data nodes in the server model and aggregate them
        """
        weights = []
        for data_node in self._federated_data:
            weights.append(data_node.query_model_params())

        aggregated_weights = self._aggregator.aggregate_weights(weights)

        return aggregated_weights

    @abc.abstractmethod
    def run_rounds(self, test_data, test_label):
        """
        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            test_data: Test data for evaluation between rounds
            test_label: Test label for evaluation between rounds
        """
