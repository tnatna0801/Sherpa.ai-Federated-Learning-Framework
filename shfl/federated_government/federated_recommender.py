import abc
import numpy as np

from shfl.federated_government.distributed_government import DistributedGovernment


class FederatedRecommender(DistributedGovernment):
    """
    Class used to represent the central class FederatedRecommender.

    # Arguments:
       model_builder: Function that return a trainable model (see: [Model](../model))
       federated_data: Federated data to use.
       (see: [FederatedData](../private/federated_operation/#federateddata-class))
       aggregator: Federated aggregator function (see: [Federated Aggregator](../federated_aggregator))
       model_param_access: Policy to access model's parameters, by default non-protected
       (see: [DataAccessDefinition](../private/data/#dataaccessdefinition-class))

    # Properties:
        global_model: Return the global model.

    The training and test data are arrays such that the first column denotes the clientId to which that data belongs.
    Every model in a node must contain the clientId property.
    """

    def __init__(self, model_builder_server, model_builder_node, federated_data, aggregator, model_params_access=None):
        super().__init__(model_builder_server, model_builder_node, federated_data, aggregator, model_params_access)

    def evaluate_global_model(self, data_test, label_test):
        """
        Method that evaluates the performance of the global model.

        # Arguments:
            test_data: test dataset. The first column corresponds to the clientId.
            test_label: corresponding labels to test dataset
        """
        evaluations, _ = self.evaluate_clients(data_test, label_test)
        return self.evaluate_global_model_recommender(data_test, label_test, evaluations)

    @abc.abstractmethod
    def evaluate_global_model_recommender(self, data_test, label_test, evaluations):
        """
        Method that evaluates the performance of the global model using the evaluation of each client.

        # Arguments:
            test_data: test dataset. The first column corresponds to the clientId.
            test_label: corresponding labels to test dataset.
            evaluations: dictionary containing the evaluation of the clients.
        """

    @abc.abstractmethod
    def deploy_central_model(self):
        """
        Abstract method to deploy the global learning model to each client (node) in the simulation.
        """

    def evaluate_clients(self, data_test, label_test):
        """
        Method that must implement every federated government extending this class.

        It evaluates the local learning models over global test dataset.

        # Arguments:
            test_data: test dataset. The first column corresponds to the clientId.
            test_label: corresponding labels to test dataset.

        # Returns:
            evaluations: a dictionary containing the evaluations of the clients.
            evaluations: a dictionary containing the local evaluations of the clients.
        """
        evaluations = dict.fromkeys(np.unique(data_test[:, 0]))
        local_evaluations = dict.fromkeys(np.unique(data_test[:, 0]))
        for data_node in self._federated_data:
            client_id = data_node.federated_data_identifier
            data_client = data_test[data_test[:, 0] == client_id]
            label_client = label_test[data_test[:, 0] == client_id]
            evaluations[client_id], local_evaluations[client_id] = data_node.evaluate(data_client, label_client)
        return evaluations, local_evaluations

    @abc.abstractmethod
    def run_rounds(self, test_data, test_label):
        """
        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            test_data: Test data for evaluation between rounds. The first column corresponds to the clientId.
            test_label: Test label for evaluation between rounds
        """
