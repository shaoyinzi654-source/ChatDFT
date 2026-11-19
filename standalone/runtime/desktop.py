"""Per-user desktop window activation for repeated launches."""
from hashlib import sha256

from PySide6.QtCore import QObject, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from chatdft.storage import data_dir

REQUESTS = {b'show', b'server-jobs', b'pdc-models', b'pdc-and-server'}


def server_name():
    identity = str(data_dir().resolve()).casefold().encode('utf-8')
    return 'ChatDFT-Standalone-5.8-' + sha256(identity).hexdigest()[:20]


def activate_existing(args):
    socket = QLocalSocket()
    socket.connectToServer(server_name())
    if not socket.waitForConnected(250):
        return False
    if args.pdc_models:
        request = b'pdc-and-server' if args.server_jobs else b'pdc-models'
    else:
        request = b'server-jobs' if args.server_jobs else b'show'
    socket.write(request + b'\n')
    socket.flush()
    if not socket.waitForReadyRead(1500):
        socket.abort()
        return False
    return bytes(socket.readAll()).strip() == b'OK'


class InstanceServer(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.UserAccessOption)
        self.server.newConnection.connect(self.accept)
        self.server.listen(server_name())

    def accept(self):
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            socket.disconnected.connect(socket.deleteLater)
            socket.readyRead.connect(lambda s=socket: self.receive(s))
            if socket.bytesAvailable():
                self.receive(socket)

    def receive(self, socket):
        if socket.bytesAvailable() > 32:
            socket.abort()
            return
        if not socket.canReadLine():
            return
        request = bytes(socket.readLine()).strip()
        if request not in REQUESTS:
            socket.abort()
            return
        window = self.window
        if window.isMinimized():
            window.showNormal()
        window.raise_()
        window.activateWindow()
        socket.write(b'OK\n')
        socket.disconnectFromServer()
        if request in (b'server-jobs', b'pdc-and-server'):
            QTimer.singleShot(0, window.open_server_jobs)
        if request in (b'pdc-models', b'pdc-and-server'):
            QTimer.singleShot(0, window.prepare_pdc_paper)
