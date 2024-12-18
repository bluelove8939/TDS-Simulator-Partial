class CustomException(Exception):
    def __init__(self, inst: any, msg: str, *args: object) -> None:
        super().__init__(*args)
        
        self.inst = inst
        self.msg = msg
        
    def __str__(self) -> str:
        return f"[ERROR] {type(self.inst).__name__}: {self.msg}"
    

class CustomAPIException(Exception):
    def __init__(self, inst: str, msg: str, *args: object) -> None:
        super().__init__(*args)
        
        self.inst = inst
        self.msg = msg
        
    def __str__(self) -> str:
        return f"[API ERROR] {self.inst}: {self.msg}"


class ReturnCode(object):
    def __init__(self, code: int) -> None:
        self.code = code
        self.msg  = ""
    
    def __bool__(self) -> int:
        return self.code
    
    def __str__(self) -> str:
        return f"ReturnCode(code={self.code})"

class ErrorReturnCode(ReturnCode):
    def __init__(self, msg: str="") -> None:
        super().__init__(code=False)
        
        self.msg = msg
    
    def __str__(self) -> str:
        return f"ErrorCode(msg=\"{self.msg}\")"
        