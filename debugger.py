"""Simulation commands run on the simulation thread at bounded safe points."""

from constants import BAIE_STAGNATION_THRESHOLD, PATH_NAMES, TITAN_NAMES
from console_control import ConsoleInput
import logging

logger = logging.getLogger("OmphalosLogger")


class Debugger:
    def __init__(self, simulation, interactive=None):
        self.sim = simulation
        self.paused = False
        self.last_command = ""
        self.stop_requested = False
        self.stop_signal = None
        self._console = ConsoleInput(interactive)

    @property
    def interactive(self):
        return self._console.interactive

    def request_stop(self, signum=None):
        """Signal-safe request: no I/O or locks; the main loop handles saving."""
        if signum is not None and self.stop_signal is None:
            self.stop_signal = signum
        self.stop_requested = True

    def poll(self):
        """Read hotkeys only when running; command input has the same owner."""
        if self.paused or self.stop_requested:
            return
        for _ in range(64):
            char = self._console.read_char()
            if char is None:
                return
            if char in ("\x04", "\x1a"):
                self.close()
                return
            if char.lower() == "p":
                self.paused = True
                return

    def close(self):
        self._console.close()

    def handle_commands(self):
        """Wait in short intervals so a stop signal also works while paused."""
        if self.stop_requested:
            return
        if not self.interactive:
            self.paused = False
            return
        prompt = "\033[93m(翁法罗斯创世涡心) > \033[0m"
        self._console.write(prompt)
        line = []
        while self.paused and not self.stop_requested:
            char = self._console.read_char(timeout=0.05)
            if not self.interactive or char in ("\x04", "\x1a"):
                # EOF must neither spin nor strand a headless simulation paused.
                self.close()
                self.paused = False
                return
            if char is None:
                continue
            if char in ("\r", "\n"):
                self._console.write("\n")
                try:
                    self._execute_command("".join(line).strip())
                except (ValueError, TypeError, OSError) as exc:
                    logger.info("命令执行失败: %s", exc)
                line.clear()
                if self.paused and not self.stop_requested:
                    self._console.write(prompt)
            elif char in ("\b", "\x7f"):
                if line:
                    line.pop()
                    self._console.write("\b \b")
            elif char.isprintable():
                line.append(char)
                self._console.write(char)

    def _execute_command(self, command_line_str):
        if command_line_str.lower().split()[:1] == ["stop"]:
            self.request_stop()
            self.paused = False
            return
        if not command_line_str:
            return
        command_line = command_line_str.split()
        cmd = command_line[0].lower()
        args = command_line[1:]
        if cmd in ('c', 'continue'):
            self.paused = False
            return
        elif cmd in ('n', 'next'):
            self.last_command = 'next'
            self.paused = False
            return
        elif cmd in ('p', 'print'):
            if not args:
                logger.info("错误: 请提供实体名称或 'baie'。用法: p <name|baie>")
                return
            entity_name = ' '.join(args)
            entity = None
            if entity_name.lower() == 'baie':
                entity = self.sim.reincarnator
                if not entity:
                    logger.info('错误: 当前没有卡厄斯兰那实体。')
                    return
            else:
                entity = self.sim.name_to_entity_map.get(entity_name)
            if entity:
                logger.info(f'\n--- 实体详情: {entity.name} ---')
                logger.info(entity)
                logger.info('  泰坦亲和度:')
                for i, name in enumerate(TITAN_NAMES):
                    logger.info(f'    {name:<4}: {entity.titan_affinities[i]:.2f}')
                logger.info('  命途倾向:')
                for i, name in enumerate(PATH_NAMES):
                    logger.info(f'    {name:<4}: {entity.path_affinities[i]:.3f}')
                logger.info('---')
            else:
                logger.info(f"错误: 未找到名为 '{entity_name}' 的实体。")
        elif cmd == 'top':
            k = int(args[0]) if args and args[0].isdigit() else 5
            sorted_pop = sorted(self.sim.population, key=lambda p: p.score, reverse=True)
            top_k = sorted_pop[:k]
            logger.info(f'\n--- 当前评分 Top {k} ---')
            for i, p in enumerate(top_k):
                logger.info(f'{i + 1}. {p}')
            logger.info('---')
        elif cmd == 'status':
            diversity = 0
            if self.sim.population:
                diversity = len(set((p.dominant_path_idx for p in self.sim.population))) / len(PATH_NAMES)
            logger.info('\n--- 翁法罗斯状态报告 ---')
            logger.info(f'  世代: {self.sim.generation}/{self.sim.total_generations}')
            logger.info(f'  种群数量: {len(self.sim.population)}')
            logger.info(f'  生态多样性: {diversity:.2%}')
            logger.info(f'  当前突变率: {self.sim.population_manager.mutation_rate:.4f}')
            if self.sim.stagnation_manager.long_term_stagnation_counter:
                logger.info(f'  全局停滞计数: {self.sim.stagnation_manager.long_term_stagnation_counter} / 10 (触发唤醒)')
            if self.sim.reincarnator:
                logger.info(f'  白厄停滞计数: {self.sim.stagnation_manager.baie_stagnation_counter} / {BAIE_STAGNATION_THRESHOLD}')
            logger.info('---')
        elif cmd == 'zeitgeist':
            logger.info('\n--- 当前翁法罗斯思潮 ---')
            zeitgeist_status = sorted(zip(PATH_NAMES, self.sim.cosmic_zeitgeist), key=lambda item: item[1], reverse=True)
            for name, weight in zeitgeist_status:
                logger.info(f'  {name:<4}: {weight:+.4f}')
            logger.info('---')
        elif cmd == 'blueprint':
            logger.info('\n--- 当前演化蓝图 ---')
            blueprint_status = sorted(zip(TITAN_NAMES, self.sim.base_titan_affinities), key=lambda item: item[1], reverse=True)
            for name, affinity in blueprint_status:
                logger.info(f'  {name:<4}: {affinity:.4f}')
            logger.info('---')
        elif cmd == 'set':
            if len(args) != 2:
                logger.info('错误: 用法: set <parameter_name> <value>')
                return
            param, value = (args[0], args[1])
            if hasattr(self.sim, param):
                try:
                    current_val = getattr(self.sim, param)
                    setattr(self.sim, param, type(current_val)(value))
                    logger.info(f"成功: 参数 '{param}' 已被设置为 {value}。")
                except (ValueError, TypeError):
                    logger.info(f"错误: 无法将 '{value}' 转换为 '{param}' 所需的类型。")
            else:
                logger.info(f"错误: 模拟中不存在名为 '{param}' 的参数。")
        elif cmd == 'save':
            if not args:
                logger.info('错误: 请提供存档文件名。用法: save <filename.json>')
                return
            filepath = args[0]
            self.sim.save_simulation_state(filepath)
        elif cmd == 'load':
            logger.info('JSON 快照不含完整运行状态，当前不能恢复运行；现有世界未改变。')
        elif cmd == 'help':
            logger.info('\n--- 可用命令 ---')
            logger.info('  c, continue         : 继续模拟')
            logger.info('  n, next             : 执行一步并暂停（世代或有机模型训练步）')
            logger.info('  p, print <name|baie>: 打印指定实体或当前卡厄斯兰那的详细信息')
            logger.info('  top [k]             : 显示评分最高的k个实体 (默认 k=5)')
            logger.info('  status              : 显示当前的翁法罗斯宏观状态')
            logger.info('  zeitgeist           : 查看当前的翁法罗斯思潮权重')
            logger.info('  blueprint           : 查看当前的演化蓝图亲和度')
            logger.info('  set <param> <value> : 动态设置一个模拟参数 (如: set mutation_rate 0.5)')
            logger.info('  save <file.json>    : 将当前模拟状态保存到文件')
            logger.info('  load <file.json>    : 当前不支持从不完整 JSON 快照恢复运行')
            logger.info('  stop                : 保存当前状态并退出')
            logger.info('  help                : 显示此帮助信息')
            logger.info('---')
        else:
            logger.info(f"错误: 未知命令 '{cmd}'。")
