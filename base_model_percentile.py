import mesa
import random
import matplotlib.pyplot as plt
import statistics
import numpy as np


# This ABM instance should ideally reflect the behaviour of the matching algorithm 
# as well as the reputation system. 
# 
# - Potentially we would have to simplify the matching algo a bit, while we can implement the rep system 1:1
# 
# - I suppose the most interesting aspect is to see whether there is some kind of dominance of a handful of processors which emerges 


class MarketPlace(mesa.Agent):

    def __init__(self, unique_id, model):
        self.avg_reward = 0
        self.total_rewards = 0
        self.total_jobs_matched = 0
        self.total_jobs_fulfilled = 0
        self.total_jobs_failed = 0 
        self.ninetieth_percentile = 0 # the ninetieth percentile with respect to processors' reputations
        self.rewards = [] 
        self.jobs_to_be_matched = {} # indexed by consumer 
        self.advertisements = {} # indexed by processor
        super().__init__(unique_id, model)
        
    def register_job(self, job):
        if (not self.match_job(job)):
            self.jobs_to_be_matched[job['consumer_id']] = job 
        
    def register_advertisement(self, advertisement):
        self.advertisements[advertisement["processor_id"]] = advertisement

    def match_job(self, job): 
        consumer = self.model.consumers[job["consumer_id"]]
        matches = []
        def has_capacity(advertisement):
            return advertisement["capacity"] > 1 

        def is_within_budget(advertisement):
            if (job["reward"] > advertisement["fee_per_job"] ): 
                return True

        def has_min_reputation(advertisement):
            matched_processor = self.model.processors[advertisement["processor_id"]]
            return matched_processor.get_reputation() >= self.ninetieth_percentile
        
        ads_with_capacity = [ad for ad in list(self.advertisements.values()) if has_capacity(ad)]
        ads_within_budget = [ad for ad in ads_with_capacity if is_within_budget(ad)]

        if job['min_reputation']:
            matches = [ad for ad in ads_within_budget if has_min_reputation(ad)] # TODO
        else: 
            matches = ads_within_budget
            
        if len(matches) > 0: 
            random.shuffle(matches)
            slots = job['slots']
            candidates = [self.model.processors[m["processor_id"]] for m in matches][:slots]

            if (len(candidates) == slots):
                for candidate in candidates: 
                    candidate.process_job(job, self.avg_reward, slots)
                # for advertisement in matches: 
                #     self.advertisements.pop(advertisement["processor_id"]) # ❗️❗️❗️ KEY DIFFERENCE in comparison to base model. Here, we allow a processor to be allocated multiple jobs according to his capacity 
                
                consumer.on_process_job()
                self.total_jobs_matched += 1
                self.total_rewards += job['reward']
                self.avg_reward = self.total_rewards / self.total_jobs_matched 
                if (random.uniform(0, 1) > 0.9): 
                    self.ninetieth_percentile = np.percentile([p.reputation for p in self.model.processors.values()], 90)
                return True
        
            else: 
                return False            
        else: 
            return False
        
    def step(self):
        for job in self.jobs_to_be_matched.values(): 
            self.match_job(job)

class ConsumerAgent(mesa.Agent):
    has_open_jobs = False
    job = {}
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cpu_seconds = random.randint(1, 10)
        self.max_fee_per_job = random.randint(75, 125)
        self.reward = 0.5 * self.max_fee_per_job
        self.slots = 1 #random.randint(1, 10)
        self.has_multiple_slots = random.uniform(0, 1) > 0.5 

    def on_process_job(self):
        self.has_open_jobs = False

    def step(self):
        if (self.reward * 1.1 <= self.max_fee_per_job):
            self.reward *= 1.1
        job = { 
            "reward": self.reward * self.slots if self.has_multiple_slots else self.reward, 
            "min_reputation": True if random.uniform(0, 1) < 0.9 else None,
            "slots": self.slots if self.has_multiple_slots else 1, 
            "consumer_id": self.unique_id
        }
        self.model.market_place.register_job(job) 
            
class ProcessorAgent(mesa.Agent):
    weights = []
    has_open_advertisement = False
    income = 0 
    processed_jobs = 0 
    
    advertisement = {}
    r = 0 
    s = 0 
    reputation = 0

    # type ∈ [low, medium, high] depending on success_rate
    def __init__(self, success_rate, type, unique_id, model):
        super().__init__(unique_id, model)
        self.min_fee_per_job = random.randint(50, 100)
        self.fee_per_job = 2 * self.min_fee_per_job
        self.capacity = random.randint(10, 1000)

        self.success_rate = success_rate
        self.type = type

    def update_reputation(self, job, avg_reward, is_fulfilled):
        w = job['reward'] / (job['reward'] + avg_reward)
        self.weights.append(w)
        λ = self.model.lmbda
        threshold = (self.r - self.r*λ + self.s*λ)/(self.s+1)
        if is_fulfilled:      
            self.model.market_place.total_jobs_fulfilled += 1
            self.r *= λ 
            self.s *= λ 
            self.r += w
        else: 
            self.r *= λ 
            self.s *= λ    
            self.s += w   
            self.model.market_place.total_jobs_failed += 1

        self.reputation = self.calculate_reputation()


    def get_reputation(self): 
        λ = self.model.lmbda
        return self.reputation * (3-2*λ)

    def get_weights(self): 
        return sum(self.weights)/len(self.weights)

    def process_job(self, job, avg_reward, slots = None):
        self.processed_jobs += 1
        self.capacity -= 1
        self.has_open_advertisement = False
        self.model.market_place.rewards.append(job['reward'])
        is_fulfilled = random.uniform(0, 1) < self.success_rate
        if is_fulfilled: 
            if slots > 1: 
                self.income += job['reward']/slots
            else: 
                self.income += job['reward']
            self.update_reputation(job, avg_reward, True)
        else: 
            self.income -= job['reward'] 
            self.update_reputation(job, avg_reward, False) 
        self.register()

    def calculate_reputation(self):
        return (self.r+1) / (self.r+self.s+2)
         
    def get_income(self): 
        return self.income

    def get_processed_jobs(self): 
        return self.processed_jobs

    def register(self): 
        self.advertisement = {
            "fee_per_job": self.fee_per_job, 
            "capacity": self.capacity, 
            "processor_id": self.unique_id
        }
        self.model.market_place.register_advertisement(self.advertisement)
            

    def step(self):
        if self.has_open_advertisement:
            if self.fee_per_job * 0.9 > self.min_fee_per_job: 
                self.fee_per_job *= 0.9
                self.register()

        if not self.has_open_advertisement:
            self.has_open_advertisement = True
            self.register()
        
def compute_gini(processor_wealths): 
    x = sorted(processor_wealths)
    N = len(processor_wealths)
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B

class ReputationModel(mesa.Model):
    success_rates = [0.8, 0.9, 0.999]

    types = ['low', 'medium', 'high']

    def __init__(self, M, N):

        self.num_processors = M
        self.num_consumers = N

        self.lmbda = 0.98

        self.schedule = mesa.time.BaseScheduler(self)

        self.consumers = {}
        self.processors = {}

        # introduce three kinds of processors with different success_rates 
        for i in range(len(self.success_rates)): 
            for j in range(int(M/len(self.success_rates))): 
                 unique_id = "P_" + self.types[i] + '_' + str(j)
                 processor_agent = ProcessorAgent(self.success_rates[i], self.types[i], unique_id, self)
                 self.schedule.add(processor_agent)
                 self.processors[unique_id] = processor_agent
      
        for i in range(self.num_consumers):
            unique_id = "C_"+str(i)
            consumer_agent = ConsumerAgent("C_"+str(i), self)
            self.consumers[unique_id] = consumer_agent
            self.schedule.add(consumer_agent)

        self.market_place = MarketPlace(1, self)
        self.schedule.add(self.market_place)


    def step(self):
        self.schedule.step()

##################################################################################################

incomes_by_type = {}
reputations_by_type = {}
processed_jobs_by_type = {}
weights_by_type = {}

types = ['low', 'medium', 'high']

failure_rates = []
avg_reward = []
total_rewards = []
total_jobs_matched = []
for i in range(10): 
    model = ReputationModel(300, 1000) # TODO 
    for j in range(200):
        model.step()

    for type in model.types: 
        processors_by_type = [p if p.type == type else None for p in model.processors.values()]
        incomes = [p.get_income() for p in processors_by_type if p is not None]
        reputations = [p.get_reputation() for p in processors_by_type if p is not None]
        processed_jobs = [p.get_processed_jobs() for p in processors_by_type if p is not None]

        if type in incomes_by_type: 
            incomes_by_type[type] += (incomes)
        else: 
            incomes_by_type[type] = incomes
        if type in reputations_by_type: 
            reputations_by_type[type] += (reputations)
        else: 
            reputations_by_type[type] = reputations

        if type in processed_jobs_by_type: 
            processed_jobs_by_type[type] += (processed_jobs)
        else: 
            processed_jobs_by_type[type] = processed_jobs

    avg_prices = sum(model.market_place.rewards)/model.market_place.total_jobs_matched

    avg_reward.append(avg_prices)
    failure_rate = model.market_place.total_jobs_failed/(model.market_place.total_jobs_failed + model.market_place.total_jobs_fulfilled)
    failure_rates.append(failure_rate)
    total_rewards.append(model.market_place.total_rewards)
    total_jobs_matched.append(model.market_place.total_jobs_matched)

colors = ['blue','red','green']


fig = plt.figure(figsize = (10, 5))

for i, type in enumerate(types): 
    incomes = incomes_by_type[type]
    processed_jobs = processed_jobs_by_type[type]
    reputations = reputations_by_type[type]



    print('############### ' + type + ' ###############')
    print('\n')

    print('MAX INC ' + str(max(incomes)))
    print('MIN INC ' + str(min(incomes)))
    print('AVG INC ' + str(sum(incomes)/len(incomes)))
    print('STDev INC ' + str(statistics.stdev(incomes)))

    print('--------------------------')
    print('MAX REP ' + str(max(reputations)))
    print('MIN REP ' + str(min(reputations)))
    print('AVG REP ' + str(sum(reputations)/len(reputations)))
    print('STDev REP ' + str(statistics.stdev(reputations)))
    print('\n')


    plt.scatter(incomes, reputations, c = colors[i], s=1)

print('\n')
print('AVG failure rate', str(sum(failure_rates)/len(failure_rates)))
print('\n')
print('AVG PRICES', str(sum(avg_reward)/len(avg_reward)))
print('\n')
print('AVG total_rewards', str(sum(total_rewards)/len(total_rewards)))
print('\n')
print('AVG total_jobs_matched', str(sum(total_jobs_matched)/len(total_jobs_matched)))
print('\n')
print('total_jobs_matched', total_jobs_matched)
print('\n')


plt.savefig('PLOT_base_percentile.pdf')

# plt.plot(range(0, len(ginis)), ginis)
# plt.savefig('ABM_plot')
# gini = compute_gini(all_incomes)
# print(gini)
# print('avg Gini', sum(ginis)/len(ginis))
# plt.xlabel("Reward Income")
# plt.ylabel("Number of Processors")
# plt.title("Students enrolled in different courses")
# plt.hist(all_incomes, bins=range(max(all_incomes) + 1))
# plt.savefig('ABM_plot')




#❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️
# KEY Learnings
# - the impact of the reputation system depends entirely on how the matching is implemented
# 
# There are obviously many ways to implement the matching.

# It seems like there are two main approaches: 
# 
# 1)    - consider all ads which fulfill min_reputation
#       - select the cheapest
#       - in this scenario the consumer may pay less than he's willing to pay
# 
# 2)    - find all ads which are within the budget of the job
#       - select the one with the highest reputation
#       - since we're not maximizing for price, the consumer should be expected to pay more on average 
#❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️❗️
