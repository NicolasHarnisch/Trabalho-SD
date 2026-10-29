import argparse
import threading
from concurrent import futures

import grpc
import numpy as np

import federated_pb2
import federated_pb2_grpc
from common_fl import (
    aggregate_fedavg,
    create_model_template,
    evaluate_global_model,
    get_model_params,
    load_all_test_sets,
)


# Dica para os alunos:
# O servidor tem três papéis principais:
# 1) disponibilizar o modelo global atual aos clientes;
# 2) receber atualizações locais enviadas pelos clientes;
# 3) consolidar uma rodada quando todos os clientes esperados responderem.


class FederatedLearningServicer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self, num_clients: int, total_rounds: int):
        self.num_clients = num_clients
        self.total_rounds = total_rounds
        self.current_round = 1
        self.lock = threading.Lock()
        self.received_updates = {}
        self.training_finished = threading.Event()

        # O modelo global começa com um template vazio/inicial.
        self.model_template = create_model_template(num_clients)
        self.global_params = get_model_params(self.model_template)
        self.X_test_global, self.y_test_global = load_all_test_sets(num_clients)

    def _params_to_proto(self, params):
        """
        Converte [coef, intercept] para ModelParameters.

        Dica:
        - mesma lógica do cliente em params_to_proto
        - servidor e cliente devem serializar no mesmo formato
        """
        # =====================================================================
        # [IMPLEMENTAÇÃO - Passo 1] Serialização do modelo global para Protobuf
        # =====================================================================
        coef, intercept = params
        return federated_pb2.ModelParameters(
            coef_values=coef.ravel().tolist(),
            coef_shape=list(coef.shape),
            intercept_values=intercept.ravel().tolist(),
        )

    def _proto_to_params(self, proto_params):
        """
        Converte ModelParameters para [coef, intercept].

        Dica:
        - mesma lógica do cliente em params_from_proto
        """
        # =====================================================================
        # [IMPLEMENTAÇÃO - Passo 2] Desserialização dos parâmetros para NumPy
        # =====================================================================
        coef = np.array(proto_params.coef_values, dtype=np.float64).reshape(proto_params.coef_shape)
        intercept = np.array(proto_params.intercept_values, dtype=np.float64)
        return [coef, intercept]

    def GetGlobalModel(self, request, context):
        """
        Retorna a rodada atual e o modelo global atual.

        Dica:
        - done deve ser True quando current_round > total_rounds
        - a mensagem pode ser algo como "Treinamento encerrado" ou "Modelo global disponível"
        """
        with self.lock:
            # =================================================================
            # [IMPLEMENTAÇÃO - Passo 3] RPC GetGlobalModel
            # done é verdadeiro se current_round ultrapassou o total_rounds
            # =================================================================
            done = self.current_round > self.total_rounds
            proto_model = self._params_to_proto(self.global_params)
            msg = "Treinamento encerrado" if done else f"Modelo global disponível para a rodada {self.current_round}"
            return federated_pb2.GlobalModel(
                round=self.current_round,
                total_rounds=self.total_rounds,
                done=done,
                message=msg,
                model=proto_model,
            )

    def _validate_update(self, request):
        """
        Verifica se uma atualização recebida do cliente é válida.

        Regras esperadas:
        1. Se o treinamento já terminou, rejeitar.
        2. Se request.round for diferente de current_round, rejeitar.
        3. Se o mesmo cliente já enviou atualização nesta rodada, rejeitar.
        4. Caso contrário aceite e mensagem "OK".
        Retorne uma tupla: (accepted: bool, message: str)
        """
        # =====================================================================
        # [IMPLEMENTAÇÃO - Passo 4] Validação das regras da rodada
        # =====================================================================
        # Regra 1: Rejeitar se o treinamento já encerrou
        if self.current_round > self.total_rounds:
            return False, "O treinamento já terminou."

        # Regra 2: Rejeitar se a rodada do update for diferente da rodada corrente
        if request.round != self.current_round:
            return False, f"Rodada {request.round} inválida. Servidor aguarda a rodada {self.current_round}."

        # Regra 3: Rejeitar updates duplicados do mesmo cliente na mesma rodada
        if request.cid in self.received_updates:
            return False, f"Atualização duplicada do cliente {request.cid} para a rodada {self.current_round}."

        # Regra 4: Aceitar update válido
        return True, "OK"

    def _consolidate_round_if_ready(self):
        """
        Consolida a rodada quando todos os clientes esperados já responderam.

        Passos esperados:
        - verificar se len(received_updates) == num_clients
        - Montar lista de updates
        - agregar com aggregate_fedavg(...)
        - avaliar o modelo global com evaluate_global_model(...)
        - imprimir métricas da rodada
        - limpar received_updates
        - avançar current_round
        - se a última rodada terminou, sinalizar training_finished
        """
        # =====================================================================
        # [IMPLEMENTAÇÃO - Passo 5] Consolidação da rodada e FedAvg
        # Só executa quando todos os clientes esperados tiverem enviado updates
        # =====================================================================
        if len(self.received_updates) != self.num_clients:
            return

        # 5.1 Monta a lista com as atualizações dos clientes
        updates = list(self.received_updates.values())

        # 5.2 Agrega os parâmetros com a média ponderada (FedAvg)
        self.global_params = aggregate_fedavg(updates)

        # 5.3 Avalia o novo modelo global agregado no conjunto de teste unificado
        loss, acc = evaluate_global_model(
            self.global_params,
            self.model_template,
            self.X_test_global,
            self.y_test_global,
        )

        # 5.4 Calcula métricas médias locais para log informativo
        avg_train_loss = sum(u["train_loss"] for u in updates) / len(updates)
        avg_train_acc = sum(u["train_acc"] for u in updates) / len(updates)

        print(f"\n--- [Servidor] Rodada {self.current_round}/{self.total_rounds} Concluída ---")
        print(f"  Treino médio local - loss: {avg_train_loss:.4f} | acc: {avg_train_acc:.4f}")
        print(f"  Modelo global      - loss: {loss:.4f} | acc: {acc:.4f}")
        print("-" * 55)

        # 5.5 Limpa o buffer de atualizações para a próxima rodada
        self.received_updates = {}

        # 5.6 Avança o número da rodada
        self.current_round += 1

        # 5.7 Se todas as rodadas foram concluídas, sinaliza término
        if self.current_round > self.total_rounds:
            self.training_finished.set()

    def SubmitUpdate(self, request, context):
        """
        Recebe a atualização de um cliente.

        Passos esperados:
        1. Validar a atualização com _validate_update
        2. Se inválida, retornar UpdateAck(accepted=False, ...)
        3. Se válida, armazenar em received_updates[request.cid]
        4. Tentar consolidar a rodada com _consolidate_round_if_ready
        5. Retornar UpdateAck(accepted=True, ...)
        """
        with self.lock:
            # =================================================================
            # [IMPLEMENTAÇÃO - Passo 6] RPC SubmitUpdate
            # =================================================================
            # 6.1 Valida a atualização conforme as regras da rodada
            accepted, message = self._validate_update(request)
            if not accepted:
                return federated_pb2.UpdateAck(
                    accepted=False,
                    message=message,
                    server_round=self.current_round,
                )

            # 6.2 Desserializa e armazena os dados do cliente no buffer
            params = self._proto_to_params(request.model)
            self.received_updates[request.cid] = {
                "params": params,
                "num_examples": request.num_examples,
                "train_loss": request.train_loss,
                "train_acc": request.train_acc,
            }

            print(
                f"[Servidor] Recebida atualização do cliente {request.cid} "
                f"na rodada {request.round} ({len(self.received_updates)}/{self.num_clients} clientes)."
            )

            # 6.3 Tenta consolidar a rodada se todos os clientes já responderam
            self._consolidate_round_if_ready()

            # 6.4 Retorna confirmação de sucesso ao cliente
            return federated_pb2.UpdateAck(
                accepted=True,
                message="OK",
                server_round=self.current_round,
            )




def serve(host: str, port: int, num_clients: int, total_rounds: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedLearningServicer(num_clients, total_rounds)
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)

    address = f"{host}:{port}"
    server.add_insecure_port(address)
    server.start()
    print(f"Servidor gRPC ouvindo em {address}")
    print(f"Aguardando {num_clients} clientes por rodada, total de {total_rounds} rodadas.\n")

    # Dica:
    # wait_for_termination() bloqueia para sempre.
    # Aqui queremos encerrar quando o treinamento terminar.
    servicer.training_finished.wait()
    print("[Servidor] Todas as rodadas foram concluídas. Encerrando servidor gRPC...")
    server.stop(grace=2).wait()
    print("[Servidor] Servidor encerrado com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    serve(args.host, args.port, args.num_clients, args.rounds)
